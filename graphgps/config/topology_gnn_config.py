"""Config registration for the active ST-PINN topology model."""

from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("topology_gnn")
def topology_gnn_cfg(cfg):
    cfg.topology_gnn = CN()
    cfg.topology_gnn.information_mode = "old_state"

    cfg.topology_gnn.hidden_dim = 128
    cfg.topology_gnn.dropout = 0.1
    cfg.topology_gnn.residual = True

    cfg.topology_gnn.num_heads = 4
    cfg.topology_gnn.num_diffusion_steps = 4
    cfg.topology_gnn.attention_every_k_steps = 1
    cfg.topology_gnn.enable_global_attn = True

    cfg.topology_gnn.inject_rho_to_edges = True
    cfg.topology_gnn.inject_flow_to_edges = True
    cfg.topology_gnn.inject_rho_to_nodes = True
    cfg.topology_gnn.initial_flow_mode = "old_flow_warm_start"
    cfg.topology_gnn.initial_pressure_mode = "from_initial_flow"
    cfg.topology_gnn.pressure_update_mode = "lwr"
    cfg.topology_gnn.share_diffusion_cell = True
    cfg.topology_gnn.alignment_mode = "full"

    cfg.topology_gnn.local_backbone = "gatedgcn"
    cfg.topology_gnn.num_edge_transformer_layers = 1
    cfg.topology_gnn.ffn_type = "relu"
    cfg.topology_gnn.ffn_mult = "4"
    cfg.topology_gnn.norm_type = "layernorm"
    cfg.topology_gnn.norm_position = "post"
    cfg.topology_gnn.edge_to_node_agg = "mean"
    cfg.topology_gnn.edge_endpoint_mode = "fusion"
    cfg.topology_gnn.init_scheme = "default"
    cfg.topology_gnn.init_residual_scale = 0.1
    cfg.topology_gnn.init_delta_scale = 0.1
