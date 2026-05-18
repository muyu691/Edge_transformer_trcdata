# Zero-shot Generalization to New Networks

This task evaluates whether a topology reconfiguration model trained on one
traffic network transfers to another network without fine-tuning.

## What Is Evaluated

The script loads a source-network checkpoint and evaluates it directly on a
target network split. The default pairs in `scripts/run_zero_shot_generalization.sh`
are:

- SiouxFalls checkpoint -> EMA test split
- EMA checkpoint -> SiouxFalls test split

For each target split it reports:

- normalized MAE/RMSE/R2/WMAPE
- denormalized real-flow MAE/RMSE/R2/WMAPE
- retained-edge and newly-added-edge metrics
- flow-conservation violation metrics: Con-MAE, Con-RMSE, Con-Max, RelCon, RelCon p95
- forward inference time and milliseconds per graph

## Important Scaler Policy

GraphGPS checkpoints save `flow_mean` and `flow_std` buffers from the source
training network. In zero-shot evaluation, the target dataset has its own flow
normalizer. The default `--scaler-policy target` resets the model buffers to the
target scaler after loading checkpoint weights.

This keeps:

- target normalized inputs and labels consistent with model physics updates
- denormalized flow metrics in the target network's physical units
- conservation residuals in the target network's units

Use `--scaler-policy checkpoint` only for debugging the effect of keeping the
source-network scaler.

## Outputs

Each run writes:

- `zero_shot_eval.log`
- `zero_shot_<source>_to_<target>.json`

By default, outputs are under:

```bash
results/zero_shot_generalization/<Source>_to_<Target>/
```

## Vera Commands

Smoke validation, one batch per transfer direction:

```bash
cd /cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main
SMOKE=1 DEVICE=auto bash scripts/run_zero_shot_generalization.sh
```

Full selected-split evaluation:

```bash
cd /cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main
DEVICE=auto bash scripts/run_zero_shot_generalization.sh
```

Run only two batches for a quick GPU sanity check:

```bash
cd /cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main
MAX_BATCHES=2 DEVICE=cuda bash scripts/run_zero_shot_generalization.sh
```

Override checkpoints or data root when Vera paths differ:

```bash
cd /cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main
DATA_ROOT=/path/to/processed_data \
SIOUX_CKPT=/path/to/siouxfalls/ckpt/175.ckpt \
EMA_CKPT=/path/to/ema/ckpt/185.ckpt \
DEVICE=auto \
bash scripts/run_zero_shot_generalization.sh
```
