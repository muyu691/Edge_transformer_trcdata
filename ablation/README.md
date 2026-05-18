# Ablation Design

This folder stores ablation study launchers that are still relevant to the
current ST-PINN mainline.

All ablations share the same base training setup unless the ablation itself changes the structure:

- dataset format: `PyG-NetworkPairs`
- hidden dim: `128`
- diffusion steps: `4`
- batch size: `32`
- optimizer: `adamW`
- base lr: `1e-3`
- weight decay: `1e-5`
- max epoch: `200`
- objective: `L = L_s + lambda_con * L_con`
- selection metric: `validation.rmse_norm`
- best checkpoint selection enabled before final test evaluation

## Variants

1. `no_virtual_links`
   - `enable_global_attn=False`
   - Purpose: remove implicit virtual routing / global attention while keeping the local GatedGCN diffusion path.

2. `no_rho_injection`
   - `inject_rho_to_edges=False`
   - `inject_flow_to_edges=True`
   - `inject_rho_to_nodes=False`
   - Purpose: remove pressure injection but keep recurrent flow-state input on edges.

3. `no_lwr_recurrence`
   - `pressure_update_mode=fixed_initial`
   - Purpose: keep the initial pressure shockwave but remove the step-by-step physical pressure evolution.

4. `unshared_diffusion_cells`
   - `share_diffusion_cell=False`
   - Purpose: replace one shared recurrent cell with `K` different cells.
   - Note: parameter count is expected to increase; compare both performance and logged parameter count / wall-clock time.

5. `new_attr_only`
   - `alignment_mode=new_attr_only`
   - Purpose: feed only `new_attr`; zero old attributes, old flow, and `is_new_edge`.

## Removed

- `diffusion_backbone_only`
  - This ablation has been removed because the current mainline no longer
    contains the old equilibrium-augmentation branch. It is now identical to
    the default model.

## Dataset Resolution

The launchers support either:

- `DATASET_DIR=/cephyr/.../processed_data/<dataset_dir>`
- `DATASET_DIR=...` together with `NETWORK_NAME=<YourNetworkName>` for custom networks
- or `DATASET_NAME=ema|siouxfalls|anaheim`

Default automatic network-name resolution:

- `ema` -> `EMA`
- `siouxfalls` -> `SiouxFalls`
- `anaheim` -> `Anaheim`

If `DATASET_DIR` is not provided, the launchers pass `PROCESSED_ROOT` to the
loader and let the loader resolve the actual processed dataset directory from
`dataset_meta.json` and `NETWORK_NAME`.
