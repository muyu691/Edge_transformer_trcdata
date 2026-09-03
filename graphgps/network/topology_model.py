from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import to_dense_batch

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.utils import match_edge_indices


class _GNNBatch:
    """Minimal container used to drive GatedGCNLayer.forward()."""

    __slots__ = ("x", "edge_attr", "edge_index")

    def __init__(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> None:
        self.x = x
        self.edge_attr = edge_attr
        self.edge_index = edge_index


class EdgeAlignmentModule(nn.Module):
    """Align old/new edge features on the new graph edge set."""

    def __init__(self, mode: str = "full") -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in ("full", "new_attr_only", "wo_old_flow"):
            raise ValueError(f"Unsupported EdgeAlignmentModule mode: {mode}")
        self.mode = mode

    def forward(
        self,
        edge_index_old: torch.Tensor,
        edge_attr_old: torch.Tensor,
        flow_old: torch.Tensor,
        edge_index_new: torch.Tensor,
        edge_attr_new: torch.Tensor,
        total_nodes: int,
    ) -> torch.Tensor:
        device = edge_attr_new.device
        dtype = edge_attr_new.dtype
        num_new_edges = edge_index_new.shape[1]

        match_idx = match_edge_indices(
            edge_index_old=edge_index_old,
            edge_index_new=edge_index_new,
            total_nodes=total_nodes,
        )

        old_feats = torch.cat([edge_attr_old, flow_old], dim=-1)
        aligned_old = torch.zeros(num_new_edges, 4, dtype=dtype, device=device)
        retained_mask = match_idx >= 0
        if retained_mask.any():
            aligned_old[retained_mask] = old_feats[match_idx[retained_mask]]

        is_new_edge = (~retained_mask).to(dtype).unsqueeze(1)
        aligned_features = torch.cat([aligned_old, edge_attr_new, is_new_edge], dim=-1)

        if self.mode == "new_attr_only":
            aligned_features = torch.cat(
                [
                    torch.zeros_like(aligned_old),
                    edge_attr_new,
                    torch.zeros_like(is_new_edge),
                ],
                dim=-1,
            )
        elif self.mode == "wo_old_flow":
            aligned_features = torch.cat(
                [
                    aligned_old[:, :3],
                    torch.zeros(num_new_edges, 1, dtype=dtype, device=device),
                    edge_attr_new,
                    is_new_edge,
                ],
                dim=-1,
            )

        return aligned_features


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the hidden dimension."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _parse_multiplier(value) -> float:
    if isinstance(value, str):
        value = value.strip()
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return float(numerator) / float(denominator)
    return float(value)


def _make_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    norm_type = str(norm_type).lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "rmsnorm":
        return RMSNorm(hidden_dim)
    raise ValueError("norm_type must be one of: 'layernorm', 'rmsnorm'")


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block with configurable expansion."""

    def __init__(self, hidden_dim: int, ffn_mult, dropout: float) -> None:
        super().__init__()
        inner_dim = max(1, int(round(hidden_dim * _parse_multiplier(ffn_mult))))
        self.gate_proj = nn.Linear(hidden_dim, inner_dim)
        self.up_proj = nn.Linear(hidden_dim, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(x))


def _make_ffn(ffn_type: str, hidden_dim: int, ffn_mult, dropout: float) -> nn.Module:
    ffn_type = str(ffn_type).lower()
    if ffn_type == "swiglu":
        return SwiGLUFFN(hidden_dim, ffn_mult=ffn_mult, dropout=dropout)

    inner_dim = max(1, int(round(hidden_dim * _parse_multiplier(ffn_mult))))
    if ffn_type == "relu":
        activation: nn.Module = nn.ReLU()
    elif ffn_type == "gelu":
        activation = nn.GELU()
    else:
        raise ValueError("ffn_type must be one of: 'relu', 'gelu', 'swiglu'")

    return nn.Sequential(
        nn.Linear(hidden_dim, inner_dim),
        activation,
        nn.Dropout(dropout),
        nn.Linear(inner_dim, hidden_dim),
        nn.Dropout(dropout),
    )


class TransformerSelfAttentionBlock(nn.Module):
    """Dense self-attention block for graph-local node or edge tokens."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = _make_norm(
            getattr(cfg.topology_gnn, "norm_type", "layernorm"),
            hidden_dim,
        )
        self.norm2 = _make_norm(
            getattr(cfg.topology_gnn, "norm_type", "layernorm"),
            hidden_dim,
        )
        self.ffn = _make_ffn(
            getattr(cfg.topology_gnn, "ffn_type", "relu"),
            hidden_dim,
            getattr(cfg.topology_gnn, "ffn_mult", 4.0),
            dropout,
        )
        self.norm_position = str(getattr(cfg.topology_gnn, "norm_position", "post")).lower()
        if self.norm_position not in ("pre", "post"):
            raise ValueError("norm_position must be one of: 'pre', 'post'")

    def _attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_mask = None
        if attn_bias is not None:
            if attn_bias.dim() != 3:
                raise ValueError("attn_bias must have shape [batch, target_len, source_len]")
            batch_size, target_len, source_len = attn_bias.shape
            attn_mask = attn_bias[:, None, :, :].expand(
                batch_size,
                self.num_heads,
                target_len,
                source_len,
            )
            attn_mask = attn_mask.reshape(batch_size * self.num_heads, target_len, source_len)

        attn_out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return self.dropout(attn_out)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_position == "pre":
            x = x + self._attention(self.norm1(x), mask, attn_bias=attn_bias)
            x = x + self.ffn(self.norm2(x))
            return x

        x = self.norm1(x + self._attention(x, mask, attn_bias=attn_bias))
        x = self.norm2(x + self.ffn(x))
        return x


class ImplicitVirtualRoutingLayer(nn.Module):
    """Global node self-attention over each graph in the mini-batch."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = TransformerSelfAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x_dense, mask = to_dense_batch(x, batch)
        x_dense = self.block(x_dense, mask)
        return x_dense[mask]


class SharedEndpointAttentionBias(nn.Module):
    """Learned edge attention bias for edge pairs that share graph endpoints."""

    def __init__(self) -> None:
        super().__init__()
        # Query/key edge endpoint relations: src-src, src-dst, dst-src, dst-dst.
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        src_dense: torch.Tensor,
        dst_dense: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_pair = edge_mask.unsqueeze(2) & edge_mask.unsqueeze(1)
        eye = torch.eye(edge_mask.shape[1], dtype=torch.bool, device=edge_mask.device)
        valid_pair = valid_pair & ~eye.unsqueeze(0)

        src_q = src_dense.unsqueeze(2)
        dst_q = dst_dense.unsqueeze(2)
        src_k = src_dense.unsqueeze(1)
        dst_k = dst_dense.unsqueeze(1)

        bias = self.bias.new_zeros(src_dense.shape[0], src_dense.shape[1], src_dense.shape[1])
        bias = bias + self.bias[0] * ((src_q == src_k) & valid_pair).to(bias.dtype)
        bias = bias + self.bias[1] * ((src_q == dst_k) & valid_pair).to(bias.dtype)
        bias = bias + self.bias[2] * ((dst_q == src_k) & valid_pair).to(bias.dtype)
        bias = bias + self.bias[3] * ((dst_q == dst_k) & valid_pair).to(bias.dtype)
        return bias


class EdgeTransformerBackbone(nn.Module):
    """Edge-token Transformer that updates edge states and derives node states."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        residual: bool,
        num_layers: int,
        edge_to_node_agg: str,
        edge_endpoint_mode: str,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)
        self.edge_to_node_agg = str(edge_to_node_agg).lower()
        if self.edge_to_node_agg not in ("mean", "sum"):
            raise ValueError("edge_to_node_agg must be one of: 'mean', 'sum'")
        edge_endpoint_mode = str(edge_endpoint_mode).lower()
        if edge_endpoint_mode in ("shared_bias", "endpoint_bias"):
            edge_endpoint_mode = "shared_endpoint_bias"
        self.edge_endpoint_mode = edge_endpoint_mode
        if self.edge_endpoint_mode not in ("fusion", "shared_endpoint_bias"):
            raise ValueError(
                "edge_endpoint_mode must be one of: 'fusion', 'shared_endpoint_bias'"
            )

        self.endpoint_proj = (
            nn.Linear(hidden_dim * 2, hidden_dim)
            if self.edge_endpoint_mode == "fusion"
            else None
        )
        self.endpoint_attention_bias = (
            SharedEndpointAttentionBias()
            if self.edge_endpoint_mode == "shared_endpoint_bias"
            else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            _make_norm(getattr(cfg.topology_gnn, "norm_type", "layernorm"), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _aggregate_edges(
        self,
        h_e: torch.Tensor,
        index: torch.Tensor,
        total_nodes: int,
    ) -> torch.Tensor:
        agg = h_e.new_zeros(total_nodes, h_e.shape[-1])
        agg.scatter_add_(0, index.unsqueeze(-1).expand(-1, h_e.shape[-1]), h_e)
        if self.edge_to_node_agg == "mean":
            counts = h_e.new_zeros(total_nodes, 1)
            counts.scatter_add_(0, index.unsqueeze(-1), h_e.new_ones(h_e.shape[0], 1))
            agg = agg / counts.clamp_min(1.0)
        return agg

    def forward(
        self,
        h_v: torch.Tensor,
        h_e: torch.Tensor,
        rho_v: torch.Tensor,
        edge_index_new: torch.Tensor,
        batch_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index_new[0], edge_index_new[1]
        if self.edge_endpoint_mode == "fusion":
            h_e = h_e + self.endpoint_proj(torch.cat([h_v[src], h_v[dst]], dim=-1))

        edge_batch = batch_vec[src]
        h_e_dense, edge_mask = to_dense_batch(h_e, edge_batch)
        attn_bias = None
        if self.edge_endpoint_mode == "shared_endpoint_bias":
            src_dense, _ = to_dense_batch(src, edge_batch, fill_value=-1)
            dst_dense, _ = to_dense_batch(dst, edge_batch, fill_value=-1)
            attn_bias = self.endpoint_attention_bias(src_dense, dst_dense, edge_mask)
            attn_bias = attn_bias.to(device=h_e_dense.device, dtype=h_e_dense.dtype)

        for layer in self.layers:
            h_e_dense = layer(h_e_dense, edge_mask, attn_bias=attn_bias)
        h_e_new = h_e_dense[edge_mask]

        total_nodes = int(h_v.shape[0])
        incoming_agg = self._aggregate_edges(h_e_new, dst, total_nodes)
        outgoing_agg = self._aggregate_edges(h_e_new, src, total_nodes)
        node_input = torch.cat([h_v, incoming_agg, outgoing_agg, rho_v], dim=-1)
        h_v_update = self.node_update(node_input)
        h_v_new = h_v + h_v_update if self.residual else h_v_update
        return h_v_new, h_e_new


class DiffusionCell(nn.Module):
    """One pseudo-time diffusion step."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        residual: bool,
        inject_rho_to_edges: bool = True,
        inject_flow_to_edges: bool = True,
        inject_rho_to_nodes: bool = True,
        local_backbone: str = "gatedgcn",
        num_edge_transformer_layers: int = 1,
        edge_to_node_agg: str = "mean",
        edge_endpoint_mode: str = "fusion",
        enable_global_attn: bool = True,
    ) -> None:
        super().__init__()
        self.inject_rho_to_edges = bool(inject_rho_to_edges)
        self.inject_flow_to_edges = bool(inject_flow_to_edges)
        self.inject_rho_to_nodes = bool(inject_rho_to_nodes)
        self.local_backbone = str(local_backbone).lower()
        if self.local_backbone not in ("gatedgcn", "edge_transformer"):
            raise ValueError("local_backbone must be one of: 'gatedgcn', 'edge_transformer'")

        edge_extra_dim = 0
        if self.inject_rho_to_edges:
            edge_extra_dim += 2
        if self.inject_flow_to_edges:
            edge_extra_dim += 1
        self.edge_inject = nn.Linear(hidden_dim + edge_extra_dim, hidden_dim)

        node_extra_dim = 1 if self.inject_rho_to_nodes else 0
        self.node_inject = nn.Linear(hidden_dim + node_extra_dim, hidden_dim)

        if self.local_backbone == "gatedgcn":
            self.local_gnn = GatedGCNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                residual=residual,
            )
            self.edge_transformer = None
        else:
            self.local_gnn = None
            self.edge_transformer = EdgeTransformerBackbone(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                residual=residual,
                num_layers=num_edge_transformer_layers,
                edge_to_node_agg=edge_to_node_agg,
                edge_endpoint_mode=edge_endpoint_mode,
            )
        self.global_attn = (
            ImplicitVirtualRoutingLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            if bool(enable_global_attn)
            else None
        )
        self.delta_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_v: torch.Tensor,
        h_e: torch.Tensor,
        rho_v: torch.Tensor,
        f_scaled_k: torch.Tensor,
        edge_index_new: torch.Tensor,
        batch_vec: torch.Tensor,
        apply_global_attn: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index_new[0], edge_index_new[1]

        edge_parts = [h_e]
        if self.inject_rho_to_edges:
            edge_parts.extend([rho_v[src], rho_v[dst]])
        if self.inject_flow_to_edges:
            edge_parts.append(f_scaled_k)
        h_e_injected = self.edge_inject(torch.cat(edge_parts, dim=-1))

        if self.inject_rho_to_nodes:
            h_v_injected = self.node_inject(torch.cat([h_v, rho_v], dim=-1))
        else:
            h_v_injected = self.node_inject(h_v)

        if self.local_backbone == "gatedgcn":
            mini = _GNNBatch(h_v_injected, h_e_injected, edge_index_new)
            mini = self.local_gnn(mini)
            h_v_new = mini.x
            h_e_new = mini.edge_attr
        else:
            h_v_new, h_e_new = self.edge_transformer(
                h_v=h_v_injected,
                h_e=h_e_injected,
                rho_v=rho_v,
                edge_index_new=edge_index_new,
                batch_vec=batch_vec,
            )

        if apply_global_attn and self.global_attn is not None:
            h_v_new = self.global_attn(h_v_new, batch_vec)

        delta_f_scaled = self.delta_readout(h_e_new)
        return h_v_new, h_e_new, delta_f_scaled


def _replace_bn_with_ln(module: nn.Module) -> None:
    for name, child in module.named_children():
        if "BatchNorm" in child.__class__.__name__:
            setattr(module, name, nn.LayerNorm(child.num_features))
        else:
            _replace_bn_with_ln(child)


def _zero_bias(module: nn.Module) -> None:
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def _init_linear(module: nn.Linear, init_type: str, weight_scale: float = 1.0) -> None:
    if init_type == "kaiming":
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
    elif init_type == "xavier":
        nn.init.xavier_uniform_(module.weight)
    else:
        raise ValueError("init_type must be one of: 'kaiming', 'xavier'")
    if weight_scale != 1.0:
        with torch.no_grad():
            module.weight.mul_(float(weight_scale))
    _zero_bias(module)


def _apply_variance_controlled_initialization(
    module: nn.Module,
    residual_scale: float,
    delta_scale: float,
) -> None:
    """Initialize projections to preserve signal scale in the diffusion loop."""
    residual_scale = float(residual_scale)
    delta_scale = float(delta_scale)
    initialized_linears = set()

    for child in module.modules():
        if isinstance(child, nn.MultiheadAttention):
            nn.init.xavier_uniform_(child.in_proj_weight)
            if child.in_proj_bias is not None:
                nn.init.zeros_(child.in_proj_bias)
            _init_linear(child.out_proj, "xavier", weight_scale=residual_scale)
            initialized_linears.add(id(child.out_proj))

    for name, child in module.named_modules():
        if isinstance(child, (nn.LayerNorm, RMSNorm)):
            if getattr(child, "weight", None) is not None:
                nn.init.ones_(child.weight)
            if getattr(child, "bias", None) is not None:
                nn.init.zeros_(child.bias)
            continue
        if not isinstance(child, nn.Linear) or id(child) in initialized_linears:
            continue

        if name.endswith("delta_readout.3"):
            _init_linear(child, "xavier", weight_scale=delta_scale)
        elif name.endswith("delta_readout.0"):
            _init_linear(child, "kaiming")
        elif name.endswith("node_update.0"):
            _init_linear(child, "kaiming")
        elif name.endswith("node_update.4") or name.endswith("down_proj"):
            _init_linear(child, "xavier", weight_scale=residual_scale)
        else:
            _init_linear(child, "xavier")


def _safe_std(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float()
    if tensor.numel() <= 1:
        return tensor.new_zeros(())
    return tensor.std(unbiased=False)


@register_network("topology_gnn")
class NetworkPairsTopologyModel(nn.Module):
    """ST-PINN diffusion-only model."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        del dim_in, dim_out
        super().__init__()

        self.register_buffer(
            "flow_mean",
            torch.tensor([cfg.dataset.flow_mean], dtype=torch.float32),
        )
        self.register_buffer(
            "flow_std",
            torch.tensor([cfg.dataset.flow_std], dtype=torch.float32),
        )

        hidden_dim = int(cfg.topology_gnn.hidden_dim)
        dropout = float(cfg.topology_gnn.dropout)
        residual = bool(cfg.topology_gnn.residual)
        num_heads = int(cfg.topology_gnn.num_heads)

        self.hidden_dim = hidden_dim
        self.K = int(cfg.topology_gnn.num_diffusion_steps)
        self.attention_every_k_steps = int(getattr(cfg.topology_gnn, "attention_every_k_steps", 1))
        self.enable_global_attn = bool(getattr(cfg.topology_gnn, "enable_global_attn", True))
        self.inject_rho_to_edges = bool(getattr(cfg.topology_gnn, "inject_rho_to_edges", True))
        self.inject_flow_to_edges = bool(getattr(cfg.topology_gnn, "inject_flow_to_edges", True))
        self.inject_rho_to_nodes = bool(getattr(cfg.topology_gnn, "inject_rho_to_nodes", True))
        self.initial_flow_mode = str(
            getattr(cfg.topology_gnn, "initial_flow_mode", "old_flow_warm_start")
        ).lower()
        self.initial_pressure_mode = str(
            getattr(cfg.topology_gnn, "initial_pressure_mode", "from_initial_flow")
        ).lower()
        self.pressure_update_mode = str(getattr(cfg.topology_gnn, "pressure_update_mode", "lwr")).lower()
        self.share_diffusion_cell = bool(getattr(cfg.topology_gnn, "share_diffusion_cell", True))
        self.alignment_mode = str(getattr(cfg.topology_gnn, "alignment_mode", "full")).lower()
        self.local_backbone = str(getattr(cfg.topology_gnn, "local_backbone", "gatedgcn")).lower()
        self.num_edge_transformer_layers = int(
            getattr(cfg.topology_gnn, "num_edge_transformer_layers", 1)
        )
        self.edge_to_node_agg = str(getattr(cfg.topology_gnn, "edge_to_node_agg", "mean")).lower()
        self.edge_endpoint_mode = str(
            getattr(cfg.topology_gnn, "edge_endpoint_mode", "fusion")
        ).lower()
        if self.edge_endpoint_mode in ("shared_bias", "endpoint_bias"):
            self.edge_endpoint_mode = "shared_endpoint_bias"
        self.ffn_type = str(getattr(cfg.topology_gnn, "ffn_type", "relu")).lower()
        self.norm_type = str(getattr(cfg.topology_gnn, "norm_type", "layernorm")).lower()
        self.norm_position = str(getattr(cfg.topology_gnn, "norm_position", "post")).lower()
        self.init_scheme = str(getattr(cfg.topology_gnn, "init_scheme", "default")).lower()
        self.init_residual_scale = float(getattr(cfg.topology_gnn, "init_residual_scale", 0.1))
        self.init_delta_scale = float(getattr(cfg.topology_gnn, "init_delta_scale", 0.1))
        self.log_variance_stats = bool(getattr(cfg.train, "log_variance_stats", False))

        if self.attention_every_k_steps < 1:
            raise ValueError("topology_gnn.attention_every_k_steps must be >= 1")
        if self.pressure_update_mode not in ("lwr", "fixed_initial"):
            raise ValueError(
                "topology_gnn.pressure_update_mode must be one of: 'lwr', 'fixed_initial'"
            )
        if self.initial_flow_mode not in ("old_flow_warm_start", "zeros"):
            raise ValueError(
                "topology_gnn.initial_flow_mode must be one of: "
                "'old_flow_warm_start', 'zeros'"
            )
        if self.initial_pressure_mode not in ("from_initial_flow", "zeros"):
            raise ValueError(
                "topology_gnn.initial_pressure_mode must be one of: "
                "'from_initial_flow', 'zeros'"
            )
        if self.alignment_mode not in ("full", "new_attr_only", "wo_old_flow"):
            raise ValueError(
                "topology_gnn.alignment_mode must be one of: "
                "'full', 'new_attr_only', 'wo_old_flow'"
            )
        if self.local_backbone not in ("gatedgcn", "edge_transformer"):
            raise ValueError(
                "topology_gnn.local_backbone must be one of: 'gatedgcn', 'edge_transformer'"
            )
        if self.num_edge_transformer_layers < 1:
            raise ValueError("topology_gnn.num_edge_transformer_layers must be >= 1")
        if self.edge_to_node_agg not in ("mean", "sum"):
            raise ValueError("topology_gnn.edge_to_node_agg must be one of: 'mean', 'sum'")
        if self.edge_endpoint_mode not in ("fusion", "shared_endpoint_bias"):
            raise ValueError(
                "topology_gnn.edge_endpoint_mode must be one of: "
                "'fusion', 'shared_endpoint_bias'"
            )
        if self.ffn_type not in ("relu", "gelu", "swiglu"):
            raise ValueError("topology_gnn.ffn_type must be one of: 'relu', 'gelu', 'swiglu'")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError("topology_gnn.norm_type must be one of: 'layernorm', 'rmsnorm'")
        if self.norm_position not in ("pre", "post"):
            raise ValueError("topology_gnn.norm_position must be one of: 'pre', 'post'")
        if self.init_scheme not in ("default", "variance_controlled"):
            raise ValueError(
                "topology_gnn.init_scheme must be one of: 'default', 'variance_controlled'"
            )

        self.aligner = EdgeAlignmentModule(mode=self.alignment_mode)
        self.edge_init_proj = nn.Linear(8, hidden_dim)

        if self.share_diffusion_cell:
            self.diffusion_cell = DiffusionCell(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                residual=residual,
                inject_rho_to_edges=self.inject_rho_to_edges,
                inject_flow_to_edges=self.inject_flow_to_edges,
                inject_rho_to_nodes=self.inject_rho_to_nodes,
                local_backbone=self.local_backbone,
                num_edge_transformer_layers=self.num_edge_transformer_layers,
                edge_to_node_agg=self.edge_to_node_agg,
                edge_endpoint_mode=self.edge_endpoint_mode,
                enable_global_attn=self.enable_global_attn,
            )
            self.diffusion_cells = None
        else:
            self.diffusion_cell = None
            self.diffusion_cells = nn.ModuleList(
                [
                    DiffusionCell(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        residual=residual,
                        inject_rho_to_edges=self.inject_rho_to_edges,
                        inject_flow_to_edges=self.inject_flow_to_edges,
                        inject_rho_to_nodes=self.inject_rho_to_nodes,
                        local_backbone=self.local_backbone,
                        num_edge_transformer_layers=self.num_edge_transformer_layers,
                        edge_to_node_agg=self.edge_to_node_agg,
                        edge_endpoint_mode=self.edge_endpoint_mode,
                        enable_global_attn=self.enable_global_attn,
                    )
                    for _ in range(self.K)
                ]
            )

        _replace_bn_with_ln(self)
        if self.init_scheme == "variance_controlled":
            _apply_variance_controlled_initialization(
                self,
                residual_scale=self.init_residual_scale,
                delta_scale=self.init_delta_scale,
            )

    def _project_initial_flow(self, batch) -> torch.Tensor:
        device = batch.flow_old.device
        dtype = batch.flow_old.dtype
        num_new_edges = batch.edge_index_new.shape[1]

        scaled_zero = (-self.flow_mean / self.flow_std).item()
        f_scaled_0 = torch.full((num_new_edges, 1), scaled_zero, device=device, dtype=dtype)
        if self.initial_flow_mode == "zeros":
            return f_scaled_0

        total_nodes = int(batch.num_nodes)
        match_idx = match_edge_indices(
            edge_index_old=batch.edge_index_old,
            edge_index_new=batch.edge_index_new,
            total_nodes=total_nodes,
        )
        retained_mask = match_idx >= 0
        if retained_mask.any():
            f_scaled_0[retained_mask] = batch.flow_old[match_idx[retained_mask]]
        return f_scaled_0

    def _compute_pressure_from_scaled_flow(self, f_scaled: torch.Tensor, batch) -> torch.Tensor:
        device = f_scaled.device
        dtype = f_scaled.dtype
        total_nodes = int(batch.num_nodes)

        f_real = f_scaled * self.flow_std + self.flow_mean
        f_real_flat = f_real.squeeze(-1)

        src = batch.edge_index_new[0]
        dst = batch.edge_index_new[1]
        inflow = torch.zeros(total_nodes, device=device, dtype=dtype)
        outflow = torch.zeros(total_nodes, device=device, dtype=dtype)
        inflow.scatter_add_(0, dst, f_real_flat)
        outflow.scatter_add_(0, src, f_real_flat)

        net_demand = batch.net_demand.to(device=device, dtype=dtype)
        return (inflow - outflow - net_demand).unsqueeze(-1)

    def _compute_initial_pressure(self, f_scaled_0: torch.Tensor, batch) -> torch.Tensor:
        if self.initial_pressure_mode == "zeros":
            return torch.zeros(
                int(batch.num_nodes),
                1,
                device=f_scaled_0.device,
                dtype=f_scaled_0.dtype,
            )
        return self._compute_pressure_from_scaled_flow(f_scaled_0, batch)

    def forward(self, batch):
        total_nodes = int(batch.num_nodes)
        device = batch.edge_attr_new.device
        dtype = batch.edge_attr_new.dtype

        f_scaled_0 = self._project_initial_flow(batch)
        rho_v_0 = self._compute_initial_pressure(f_scaled_0, batch)
        batch.f_init_scaled = f_scaled_0
        batch.f_init_real = f_scaled_0 * self.flow_std + self.flow_mean

        aligned_features = self.aligner(
            edge_index_old=batch.edge_index_old,
            edge_attr_old=batch.edge_attr_old,
            flow_old=batch.flow_old,
            edge_index_new=batch.edge_index_new,
            edge_attr_new=batch.edge_attr_new,
            total_nodes=total_nodes,
        )
        h_e = self.edge_init_proj(aligned_features)
        h_v = torch.ones(total_nodes, self.hidden_dim, device=device, dtype=dtype)

        f_scaled_k = f_scaled_0
        rho_v_k = rho_v_0
        rho_v_history = [rho_v_0]
        delta_f_scaled_history = []

        src = batch.edge_index_new[0]
        dst = batch.edge_index_new[1]

        for step in range(self.K):
            rho_v_input = rho_v_0 if self.pressure_update_mode == "fixed_initial" else rho_v_k
            rho_v_scaled = rho_v_input / self.flow_std
            apply_global_attn = self.enable_global_attn and (((step + 1) % self.attention_every_k_steps) == 0)
            cell = self.diffusion_cell if self.share_diffusion_cell else self.diffusion_cells[step]

            h_v, h_e, delta_f_scaled = cell(
                h_v=h_v,
                h_e=h_e,
                rho_v=rho_v_scaled,
                f_scaled_k=f_scaled_k,
                edge_index_new=batch.edge_index_new,
                batch_vec=batch.batch,
                apply_global_attn=apply_global_attn,
            )

            f_scaled_k = f_scaled_k + delta_f_scaled
            if self.log_variance_stats:
                delta_f_scaled_history.append(delta_f_scaled.detach())

            if self.pressure_update_mode == "lwr":
                delta_f_real_flat = (delta_f_scaled * self.flow_std).squeeze(-1)
                delta_in = torch.zeros(total_nodes, device=device, dtype=dtype)
                delta_out = torch.zeros(total_nodes, device=device, dtype=dtype)
                delta_in.scatter_add_(0, dst, delta_f_real_flat)
                delta_out.scatter_add_(0, src, delta_f_real_flat)
                rho_v_k = rho_v_k + (delta_in - delta_out).unsqueeze(-1)
            else:
                rho_v_k = rho_v_0
            rho_v_history.append(rho_v_k)

        batch.f_diffused_scaled_final = f_scaled_k
        batch.f_diffused_real_final = f_scaled_k * self.flow_std + self.flow_mean
        batch.rho_v_diffused_final = rho_v_k
        batch.rho_v_final = rho_v_k
        batch.rho_v_history = rho_v_history
        batch.h_v_final = h_v
        batch.h_e_final = h_e
        if self.log_variance_stats:
            if delta_f_scaled_history:
                delta_stack = torch.stack([_safe_std(delta) for delta in delta_f_scaled_history])
                delta_abs_stack = torch.stack(
                    [delta.detach().float().abs().mean() for delta in delta_f_scaled_history]
                )
                batch.delta_f_scaled_std = delta_stack.mean()
                batch.delta_f_scaled_abs_mean = delta_abs_stack.mean()
            else:
                batch.delta_f_scaled_std = f_scaled_k.new_zeros(())
                batch.delta_f_scaled_abs_mean = f_scaled_k.new_zeros(())
            batch.h_e_std = _safe_std(h_e)
            batch.h_v_std = _safe_std(h_v)
            batch.f_scaled_std = _safe_std(f_scaled_k)
            batch.rho_v_std = _safe_std(rho_v_k)
            batch.rho_v_abs_mean = rho_v_k.detach().float().abs().mean()

        return f_scaled_k, batch.y
