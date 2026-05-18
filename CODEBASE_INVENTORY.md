# Codebase Inventory

Scope: this inventory covers source-like files that are relevant for maintenance and cleanup: `.py`, `.sh`, `.yaml`, `.md`, and the root training entrypoints. It intentionally excludes generated datasets, logs, checkpoints, and result JSON files unless they affect code organization.

Status labels:
- `Active-mainline`: part of the current GraphGPS/ST-PINN training pipeline.
- `Active-data`: used to generate or load the current processed datasets.
- `Active-baseline`: standalone baseline or ablation runner that still trains/evaluates code.
- `Research-branch`: optional Bayesian / SMC / proposal-flow branch; not part of the supervised ST-PINN mainline.
- `Compatibility`: kept so old scripts/configs still run, but not central to the new flow.
- `Legacy-candidate`: strong deletion candidate if you no longer need the old path.

## Root Entrypoints And Launchers

- `main.py` — `Active-mainline`. Main GraphGym/GraphGPS entrypoint: loads config, creates loaders/model/optimizer/scheduler, and dispatches to the registered train loop.
- `setup.py` — `Compatibility`. Package install metadata so `graphgps` can be imported after `pip install -e .`.
- `run_topology_common.sh` — `Active-mainline`. Shared dataset/network resolution helpers for the current topology model launchers.
- `run_topology_gnn_vera_gpu.sh` — `Active-mainline`. Main Slurm launcher for the current `topology_gnn` training job.
- `run_ema_vera1.sh` — `Compatibility`. Thin wrapper that defaults `DATASET_NAME=ema` and delegates to `run_topology_gnn_vera_gpu.sh`.
- `run_mlp_baseline_vera_gpu.sh` — `Active-baseline`. Slurm launcher for `baseline/run.py --model mlp_baseline`.
- `run_node_centric_gnn_vera_gpu.sh` — `Active-baseline`. Slurm launcher for `baseline/run.py --model NodeCentricGNN`.
- `run_single_topology_gatedgcn_vera_gpu.sh` — `Active-baseline`. Slurm launcher for `baseline/run.py --model single_topology_gatedgcn`.
- `run_heteroscedastic_baseline_vera_gpu.sh` — `Active-baseline`. Slurm launcher for the standalone irreducible-error baseline.
- `ema_generation.sh` — `Active-data`. Cluster script for generating EMA raw/processed data.
- `run_train_conditional_flow_vera_gpu.sh` — `Research-branch`. Trains the conditional RealNVP proposal used by the flow+SMC branch.
- `run_bayes_ema_vera1.sh` — `Research-branch`. Runs Bayesian importance reweighting on raw EMA pairs.
- `run_bayes_ema_array_vera1.sh` — `Research-branch`. Slurm array version of the Bayesian baseline.
- `run_bayes_oracle_sample_sweep_array_vera1_cpu.sh` — `Research-branch`. Oracle-subset sample sweep launcher.
- `run_bayes_oracle_subset_array_vera1_cpu.sh` — `Research-branch`. High-budget Bayesian oracle subset launcher.
- `run_eval_flow_smc_vera_cpu.sh` — `Research-branch`. Evaluates the learned conditional-flow proposal plus SMC.
- `run_eval_flow_smc_model_fix_vera_cpu.sh` — `Research-branch`. Fixed-SMC flow+SMC evaluation.
- `run_eval_flow_smc_model_fix_array_vera_cpu.sh` — `Research-branch`. Array version of fixed-SMC evaluation.
- `run_eval_proposal_abc_smc_array_vera_cpu.sh` — `Research-branch`. Proposal-assisted ABC-SMC evaluation launcher.
- `run_sweep_flow_smc_bridge_fix_array_vera_cpu.sh` — `Research-branch`. Bridge-repair sweep launcher.
- `run_sweep_flow_smc_model_fix_array_vera_cpu.sh` — `Research-branch`. Flow+SMC model-fix sweep launcher.
- `run_sweep_flow_smc_oracle_subset_array_vera_cpu.sh` — `Research-branch`. Oracle-subset stability sweep launcher.

## Current GraphGPS Mainline

### `graphgps/` package bootstrap

- `graphgps/__init__.py` — `Active-mainline`. Imports all subpackages so GraphGym registrations happen on `import graphgps`.
- `graphgps/network/__init__.py` — `Active-mainline`. Auto-imports all network modules in `graphgps/network`.
- `graphgps/loss/__init__.py` — `Active-mainline`. Auto-imports loss modules for GraphGym registration.
- `graphgps/train/__init__.py` — `Active-mainline`. Auto-imports train-mode modules for GraphGym registration.
- `graphgps/loader/__init__.py` — `Active-mainline`. Auto-imports dataset loaders.
- `graphgps/layer/__init__.py` — `Active-mainline`. Auto-imports custom layers.
- `graphgps/encoder/__init__.py` — `Compatibility`. Auto-imports optional encoders.
- `graphgps/head/__init__.py` — `Compatibility`. Auto-imports optional prediction heads.
- `graphgps/optimizer/__init__.py` — `Active-mainline`. Auto-imports custom optimizer/scheduler wrappers.
- `graphgps/transform/__init__.py` — `Active-mainline`. Auto-imports transform helpers.
- `graphgps/config/__init__.py` — `Active-mainline`. Auto-imports all config registration files.

### Config registration

- `graphgps/config/defaults_config.py` — `Active-mainline`. Extends GraphGym defaults; also defines many model knobs, including current loss weights and several legacy compatibility fields.
- `graphgps/config/dataset_config.py` — `Active-mainline`. Adds traffic-network dataset metadata, scaler fields, and dataset routing flags.
- `graphgps/config/topology_gnn_config.py` — `Active-mainline`. Registers the `cfg.topology_gnn` namespace for the ST-PINN / diffusion model.
- `graphgps/config/optimizers_config.py` — `Active-mainline`. Adds warmup, grad clipping, and scheduler extras used by the custom train loop.
- `graphgps/config/split_config.py` — `Active-mainline`. Defines split selection / CV behavior.
- `graphgps/config/wandb_config.py` — `Compatibility`. WandB settings; useful only if you still use WandB.
- `graphgps/config/custom_gnn_config.py` — `Compatibility`. Adds config for the generic `custom_gnn` model, not the current `topology_gnn`.
- `graphgps/config/gt_config.py` — `Compatibility`. Leftover Graph Transformer / GPS config group; comments say GT/GPS models are not included in this clean version.
- `graphgps/config/posenc_config.py` — `Compatibility`. Positional encoding config; mostly dormant unless you re-enable PE paths.
- `graphgps/config/pretrained_config.py` — `Compatibility`. Placeholder config for pretrained loading, which `main.py` explicitly stubs out.

### Loader, logging, metrics, utils

- `graphgps/loader/master_loader.py` — `Active-mainline`. Central loader registry; resolves processed traffic datasets, injects scaler stats into `cfg`, and joins train/val/test splits.
- `graphgps/loader/dataset/network_pairs_topology.py` — `Active-mainline`. In-memory dataset wrapper for processed `train_dataset.pt` / `val_dataset.pt` / `test_dataset.pt`.
- `graphgps/loader/split_generator.py` — `Active-mainline`. Generic split setup helpers used by the loader.
- `graphgps/logger.py` — `Active-mainline`. Custom logger creation and metric output; contains some OGB stubs not relevant to traffic.
- `graphgps/metric_wrapper.py` — `Active-mainline`. Flow metrics, denormalized metric handling, WMAPE / RMSE / GEH helpers.
- `graphgps/utils.py` — `Active-mainline`. Small shared utilities; importantly includes `match_edge_indices`, which the topology model uses to align old/new edges.
- `graphgps/transform/transforms.py` — `Active-mainline`. In-memory preprocessing helpers plus the edge-feature masking transform used by ablations.

### Core model, loss, training

- `graphgps/network/topology_model.py` — `Active-mainline`. The main ST-PINN / pseudo-time diffusion model. Important note: its module docstring explicitly says `OldGraphEncoder` and `NewGraphReasoner` are retained only for reference and are no longer used in the forward pass; those internal classes are the clearest in-file legacy code.
- `graphgps/layer/gatedgcn_layer.py` — `Active-mainline`. Custom GatedGCN implementation used inside the topology model and some baselines.
- `graphgps/loss/flow_conservation_loss.py` — `Active-mainline`. Current supervised-plus-physics loss dispatch for `topology_gnn`; also contains the equilibrium/reduced-cost regularizers you may remove later.
- `graphgps/train/custom_train.py` — `Active-mainline`. Registered `custom` training loop; handles loss dispatch, metrics, checkpointing, and detailed test evaluation.
- `graphgps/optimizer/extra_optimizers.py` — `Active-mainline`. AdamW registration and custom warmup schedulers used by `main.py`.

### Generic GraphGym compatibility pieces

- `graphgps/network/custom_gnn.py` — `Legacy-candidate`. Generic GraphGym GatedGCN model. Not used by the current `configs/GatedGCN/network-pairs-topology.yaml`, which sets `model.type: topology_gnn`.
- `graphgps/loss/l1.py` — `Compatibility`. Generic GraphGym regression loss for non-topology models.
- `graphgps/encoder/linear_node_encoder.py` — `Legacy-candidate`. Optional GraphGym node encoder; current topology flow disables node encoding.
- `graphgps/encoder/linear_edge_encoder.py` — `Legacy-candidate`. Optional edge encoder; current topology flow disables edge encoding.
- `graphgps/head/edge_regression.py` — `Legacy-candidate`. Generic edge-level prediction head for GraphGym models; not used by `topology_gnn`.

## Data Generation And Dataset Preparation

### New supervised dataset pipeline

- `create_sioux_data/network_registry.py` — `Active-data`. Resolves built-in network specs (`SiouxFalls`, `EMA`, `Anaheim`) and mutation policy defaults.
- `create_sioux_data/network_parser.py` — `Active-data`. Generic TNTP parser and network metadata loader.
- `create_sioux_data/generate_scenarios.py` — `Active-data`. Generates base scenarios and mutated `(G, G')` network pairs.
- `create_sioux_data/sue_solver.py` — `Active-data`. Core SUE solver / loading routines used to produce `flows_old` and `flows_new`.
- `create_sioux_data/solve_network_pairs.py` — `Active-data`. End-to-end raw pair generation pipeline: sample scenarios, solve old/new flows, and save `network_pairs_dataset.pkl`.
- `create_sioux_data/build_network_pairs_dataset.py` — `Active-data`. Converts raw solved pairs into PyG `.pt` splits and scaler metadata used by the current model.
- `create_sioux_data/utils.py` — `Active-data`. Shared utilities for free-flow times, edge indices, numeric validation, and directory setup.

### Compatibility shim inside data pipeline

- `create_sioux_data/load_sioux.py` — `Compatibility`. Backward-compatible Sioux-only loader that now delegates to the generic parser. Search shows it is referenced only by the self-test block in `sue_solver.py`, so it is a plausible cleanup target if you no longer care about that legacy helper name.

### Bayesian / SMC / proposal-flow research branch

- `create_sioux_data/train_conditional_flow_proposal.py` — `Research-branch`. CLI trainer for the conditional RealNVP proposal.
- `create_sioux_data/conditional_flow_smc.py` — `Research-branch`. Shared proposal/likelihood utilities for the flow+SMC branch.
- `create_sioux_data/eval_flow_smc.py` — `Research-branch`. Evaluates conditional-flow proposal + SMC on raw pairs.
- `create_sioux_data/sweep_flow_smc_oracle_subset.py` — `Research-branch`. Sweeps likelihood sigma / particles / MH steps on an oracle subset.
- `create_sioux_data/proposal_abc_smc.py` — `Research-branch`. Proposal-assisted ABC-SMC implementation.
- `create_sioux_data/eval_proposal_abc_smc.py` — `Research-branch`. CLI evaluation for proposal-assisted ABC-SMC.
- `create_sioux_data/bayes_importance_reweighting.py` — `Research-branch`. Standalone Bayesian baseline via importance reweighting.
- `create_sioux_data/eval_bayes_baseline.py` — `Research-branch`. CLI wrapper around the Bayesian importance-reweighting baseline.
- `create_sioux_data/eval_bayes_oracle_subset.py` — `Research-branch`. High-budget oracle subset evaluation for the Bayesian baseline.
- `create_sioux_data/plot_bayes_oracle_sample_sweep.py` — `Research-branch`. Re-runs sample-count/sigma sweeps and plots the figure.
- `create_sioux_data/benchmark_fw_runtime.py` — `Research-branch`. Benchmarks Frank-Wolfe/SUE runtime on the raw pair dataset.

### Result-merging utilities for the research branch

- `create_sioux_data/merge_bayes_chunk_results.py` — `Research-branch`. Merges chunked Bayesian baseline outputs.
- `create_sioux_data/merge_bayes_oracle_sample_sweep.py` — `Research-branch`. Merges oracle sample-sweep outputs.
- `create_sioux_data/merge_bayes_oracle_subset_chunks.py` — `Research-branch`. Merges chunked oracle-subset results.
- `create_sioux_data/merge_flow_smc_bridge_fix_array.py` — `Research-branch`. Merges per-combination bridge-fix sweep outputs.
- `create_sioux_data/merge_flow_smc_model_fix_array.py` — `Research-branch`. Merges per-combination model-fix outputs.
- `create_sioux_data/merge_flow_smc_oracle_sweep_array.py` — `Research-branch`. Merges per-combination oracle-sweep outputs.
- `create_sioux_data/merge_flow_smc_test_chunks.py` — `Research-branch`. Merges chunked full-test flow+SMC outputs.
- `create_sioux_data/merge_proposal_abc_smc_chunks.py` — `Research-branch`. Merges chunked ABC-SMC outputs.

## Standalone Supervised Baselines

- `baseline/run.py` — `Active-baseline`. Unified training/evaluation entrypoint for the standalone supervised baselines.
- `baseline/data.py` — `Active-baseline`. Dataset bundle loading and DataLoader construction for baseline training.
- `baseline/common.py` — `Active-baseline`. Small shared helpers, including edge matching and denormalization.
- `baseline/losses.py` — `Active-baseline`. Weighted supervised loss for old/new edges.
- `baseline/metrics.py` — `Active-baseline`. Regression metric accumulation for baseline runs.
- `baseline/model_common.py` — `Active-baseline`. Shared baseline components, including edge alignment logic.
- `baseline/mlp_baseline.py` — `Active-baseline`. Edge-wise MLP baseline.
- `baseline/node_centric_gnn.py` — `Active-baseline`. Node-centric old/new graph reasoning baseline.
- `baseline/single_topology_gatedgcn.py` — `Active-baseline`. Single-topology GatedGCN baseline.
- `baseline/__init__.py` — `Active-baseline`. Package marker.

## Auxiliary Analysis / Error-Floor Baselines

### Cross-residual baseline

- `cross_residual_baseline/run.py` — `Active-baseline`. One-click orchestrator for the cross-residual irreducible-error estimate.
- `cross_residual_baseline/train_split.py` — `Active-baseline`. Trains one split-specific model for the cross-residual method.
- `cross_residual_baseline/cross_evaluate.py` — `Active-baseline`. Computes the cross-residual evaluation report.
- `cross_residual_baseline/run_multi_split.py` — `Active-baseline`. Repeats the method across multiple split seeds.
- `cross_residual_baseline/config.yaml` — `Active-baseline`. Config for the cross-residual baseline.
- `cross_residual_baseline/run_cross_baseline.sh` — `Active-baseline`. Shell launcher for `run.py`.
- `cross_residual_baseline/run_multi_split_baseline.sh` — `Active-baseline`. Shell launcher for `run_multi_split.py`.
- `cross_residual_baseline/__init__.py` — `Active-baseline`. Package marker.

### Heteroscedastic irreducible-error baseline

- `irreducible_error_baseline/run.py` — `Active-baseline`. One-click training/evaluation/comparison pipeline.
- `irreducible_error_baseline/train.py` — `Active-baseline`. Training logic and config coercion.
- `irreducible_error_baseline/evaluate.py` — `Active-baseline`. Loads a checkpoint and estimates irreducible error.
- `irreducible_error_baseline/model.py` — `Active-baseline`. Standalone heteroscedastic GNN model and Gaussian NLL loss.
- `irreducible_error_baseline/compare_with_stpinn.py` — `Active-baseline`. Compares heteroscedastic estimates with ST-PINN metrics.
- `irreducible_error_baseline/config.yaml` — `Active-baseline`. Config for the heteroscedastic baseline.
- `irreducible_error_baseline/__init__.py` — `Active-baseline`. Package marker.

## Power-Law And Representation Analysis Branch

- `power_law_baseline/prepare_data.py` — `Active-baseline`. Builds deterministic fractional-data subsets for scaling-law experiments.
- `power_law_baseline/generate_experiments.py` — `Active-baseline`. Generates GraphGPS configs/manifests for power-law runs.
- `power_law_baseline/extract_mse.py` — `Active-baseline`. Extracts per-run MSE from completed experiments.
- `power_law_baseline/extrapolate.py` — `Active-baseline`. Runs Richardson extrapolation on extracted MSE values.
- `power_law_baseline/diagnose_power_law.py` — `Active-baseline`. Checks whether current experiments support a power-law interpretation.
- `power_law_baseline/estimate_noise_floor.py` — `Active-baseline`. Estimates empirical noise floor via repeated label generation.
- `power_law_baseline/feature_sufficiency_audit.py` — `Active-baseline`. Tests whether visible graph features are sufficient to pin down the target.
- `power_law_baseline/od_oracle_audit.py` — `Active-baseline`. Compares visible-only and OD-aware oracle baselines.
- `power_law_baseline/representation_utils.py` — `Active-baseline`. Shared utilities for the power-law / oracle audit branch.
- `power_law_baseline/config.yaml` — `Active-baseline`. Config for this branch.
- `power_law_baseline/run_prepare_and_generate.sh` — `Active-baseline`. Shell wrapper for preparing data and generating configs.
- `power_law_baseline/run_training_vera.sh` — `Active-baseline`. Slurm launcher for power-law training runs.
- `power_law_baseline/run_analysis_vera.sh` — `Active-baseline`. Slurm launcher for post-hoc analysis.
- `power_law_baseline/prompts/phase1_prompt.md` — `Documentation`. Prompt/instructions used in that branch.
- `power_law_baseline/prompts/phase2_prompt.md` — `Documentation`. Prompt/instructions used in that branch.
- `power_law_baseline/prompts/phase3_prompt.md` — `Documentation`. Prompt/instructions used in that branch.
- `power_law_baseline/__init__.py` — `Active-baseline`. Package marker.

## Ablation And Config Files

- `configs/GatedGCN/network-pairs-topology.yaml` — `Active-mainline`. Current main config for training `model.type: topology_gnn`.
- `ablation/ablation_common.sh` — `Active-baseline`. Shared dataset/network resolution for ablation launchers.
- `ablation/run_diffusion_backbone_only_vera_gpu.sh` — `Active-baseline`. Runs ablation that removes the later physics/equilibrium refinement pieces.
- `ablation/run_new_attr_only_vera_gpu.sh` — `Active-baseline`. Runs alignment ablation using only new-edge attributes.
- `ablation/run_no_lwr_recurrence_vera_gpu.sh` — `Active-baseline`. Disables recurrent pressure updates.
- `ablation/run_no_rho_injection_vera_gpu.sh` — `Active-baseline`. Disables rho injection into nodes/edges.
- `ablation/run_no_virtual_links_vera_gpu.sh` — `Active-baseline`. Disables global attention / virtual links.
- `ablation/run_unshared_diffusion_cells_vera_gpu.sh` — `Active-baseline`. Uses different diffusion cells per pseudo-time step.
- `ablation/README.md` — `Documentation`. Explains the ablation suite.

## Miscellaneous Utility Script

- `scripts/run_pinn_ablation.py` — `Compatibility`. Automates a lambda sweep for the older PINN-equilibrium penalty; useful only if you still keep that loss term.

## Strongest Cleanup Candidates

These are the highest-confidence deletion candidates if your goal is to keep only the current supervised `topology_gnn` flow plus whichever baselines you still actively compare against.

- `graphgps/network/custom_gnn.py`
- `graphgps/encoder/linear_node_encoder.py`
- `graphgps/encoder/linear_edge_encoder.py`
- `graphgps/head/edge_regression.py`
- `graphgps/config/custom_gnn_config.py`
- `graphgps/config/gt_config.py`
- `graphgps/config/posenc_config.py`
- `graphgps/config/pretrained_config.py`
- `scripts/run_pinn_ablation.py` if you no longer use the old PINN lambda ablation
- `create_sioux_data/load_sioux.py` if you are comfortable dropping the legacy Sioux-only helper name
- The internal `OldGraphEncoder` and `NewGraphReasoner` classes inside `graphgps/network/topology_model.py`

## Whole Branches You Can Consider Removing As Units

- `create_sioux_data/*bayes*`, `*smc*`, `train_conditional_flow_proposal.py`, and the corresponding root `.sh` launchers, if you no longer need the Bayesian / conditional-flow / ABC-SMC research branch.
- `cross_residual_baseline/` if you no longer need cross-residual irreducible-error estimation.
- `irreducible_error_baseline/` if you no longer need the heteroscedastic irreducible-error baseline.
- `power_law_baseline/` if you no longer need scaling-law / noise-floor / feature-sufficiency analysis.
- `baseline/` and the three baseline root launchers only if you are fully done with MLP / NodeCentric / single-topology comparisons.

## Notes Before Deleting

- `graphgps/__init__.py` imports every submodule for registration side effects, so deleting a file can break imports even when it is not used by the current YAML. Remove dead registrations carefully.
- `graphgps/network/topology_model.py` contains both active code and internal dead code. That file should be cleaned surgically, not deleted wholesale.
- `configs/GatedGCN/network-pairs-topology.yaml` currently still enables the equilibrium head. If you remove that code, update both the config and `graphgps/loss/flow_conservation_loss.py` together.
- The processed-data directories and result JSON files are not included here because they are artifacts, not source.
