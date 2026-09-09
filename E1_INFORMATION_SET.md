# E1 — Information-set comparison

同一个 `configs/GatedGCN/network-pairs-topology.yaml` 中的 Edge Transformer：
128 hidden / 4 diffusion steps / 4 heads / 1 edge-transformer layer / SwiGLU 8/3 /
RMSNorm pre-norm / endpoint fusion / dropout 0.1 / shared cell / no global attention。
AdamW，lr=1e-3，weight decay=1e-5，200 epochs，原 WSD 和 staged conservation，梯度裁剪 1。
三个 learned modes 均用相同 L1、lambda_old=lambda_new=1 和 conservation schedule。

## 信息边界

| Mode | Forward 获得的信息 | 初始物理流 | demand |
|---|---|---|---|
| persistence | old/new edge indices、old flow | 旧边对齐流，新边 0 | 无需 demand |
| od_only | new edge indices/attrs、OD、centroid mapping | 全 0 | OD columns − rows |
| old_state | old/new edge indices/attrs、old flow、其 divergence | 保留边旧流，新边 0 | 旧流 divergence |
| hybrid | old-state 信息加完整 OD | 保留边旧流，新边 0 | OD columns − rows |

`batch/ptr/num_nodes` 只是 batching bookkeeping；模型白名单不包含标签。
`new_edge_mask` 仅供公共 loss 分组，不进入 OD-only forward。Old-state 不建 OD encoder。
Hybrid 采用全文后续公式与科学解释一致的 **100% OD**，不模拟缺失 OD；部分 OD 需要另行明确实验定义。
OD encoder 是 outgoing/incoming 各 `Linear(C,64)+SiLU`，拼接后 `Linear(128,128)`，
加入 centroid node state。非 centroid 增量为 0。

## 1. 准备 E1 processed data（不重新生成 SUE）

VS Code PowerShell：

```powershell
cd "C:\Users\1\Desktop\Physics-Informed\Transformer_ST_PINN\edge_transformer"
conda activate graphgps
```

当前尚未转换过 PyG 时，按顺序运行：

```powershell
python -B create_sioux_data/build_network_pairs_dataset.py `
  --input_dir create_sioux_data/processed_data/siouxfalls_7000_v3 `
  --output_dir create_sioux_data/processed_data/pyg_siouxfalls_7000_e1 `
  --expected_samples 7000 --train_ratio 0.6 --val_ratio 0.2 --seed 42
```

```powershell
python -B create_sioux_data/build_network_pairs_dataset.py `
  --input_dir create_sioux_data/processed_data/ema_7000_v3 `
  --output_dir create_sioux_data/processed_data/pyg_ema_7000_e1 `
  --expected_samples 7000 --train_ratio 0.6 --val_ratio 0.2 --seed 42
```

```powershell
python -B create_sioux_data/build_network_pairs_dataset.py `
  --input_dir create_sioux_data/processed_data/anaheim_7000_roadonly `
  --output_dir create_sioux_data/processed_data/pyg_anaheim_7000_e1 `
  --expected_samples 7000 --train_ratio 0.6 --val_ratio 0.2 --seed 42
```

若已经有前版 processed 数据，必须在对应命令后加 `--split_indices`，例如：

```powershell
python -B create_sioux_data/build_network_pairs_dataset.py `
  --input_dir create_sioux_data/processed_data/siouxfalls_7000_v3 `
  --output_dir create_sioux_data/processed_data/pyg_siouxfalls_7000_e1 `
  --expected_samples 7000 `
  --split_indices create_sioux_data/processed_data/pyg_siouxfalls_7000_v3/split_indices.npz
```

EMA/Anaheim 对应旧目录同理；不要在已经成功构建同名 E1 输出后重复执行。
原 `sample_sources.json` 会一并验证，保证复用 split 对应同一批真实样本。
生成未完成的网络请先等待完成。7000 pairs 应得到 4200/1400/1400。

输出仍为 train/val/test `.pt`，另有 `[split]_od.npy`：float32 `[N,C,C]`，
通过 `open_memmap` 逐样本按同一 split order 写入。训练正 OD 均值 `od_scale`
写在 metadata，仅用于 `log1p(Q/od_scale)`，没有 OD sklearn scaler。
Data 只额外保存小型 `centroid_pos [1,C]`、`free_flow_time_new [E,1]` 和 `first_thru_node`。
容量从现有属性 scaler 还原，不保存重复完整 NetworkX graph。

Loader 以只读 mmap 在单样本 get 时复制当前 OD，batch 为 `[B,C,C]`；
合并 splits 时只合并 graph，不把 OD materialize 到 Data 列表中。
标准化参数在网络内统一、training-only、partial_fit；所有 modes 用同一 processed 目录。
原始分片每次只读一个；转换后的 PyG graph tensors 仍按原 `.pt` 格式驻留内存。

## 2. 小型本地验证（无需正式数据或 GPU）

```powershell
python -B scripts/test_sharded_dataset_builder.py
python -B scripts/test_e1_information_set.py
```

测试只在自动清理的临时目录写入微型数据，非论文结果。
也可对正式数据尝试单独两轮流程（会评价一次完整 test；不用于调参）：

```powershell
python -B scripts/e1_information_set.py --network siouxfalls --mode old_state --seed 42 --device cpu --smoke
```

`--smoke` 输出到 `results/e1_smoke`，汇总器拒绝把它计入正式 E1。

## 3. 单个正式任务

GPU 环境下：

```powershell
python -B scripts/e1_information_set.py --network siouxfalls --mode persistence --device cuda
```

```powershell
python -B scripts/e1_information_set.py --network siouxfalls --mode od_only --seed 42 --device cuda
```

```powershell
python -B scripts/e1_information_set.py --network siouxfalls --mode old_state --seed 42 --device cuda
```

```powershell
python -B scripts/e1_information_set.py --network siouxfalls --mode hybrid --seed 42 --device cuda
```

把 network 换成 `ema` / `anaheim` 即可。无 CUDA 时使用 `--device cpu`，但不要把 CPU/GPU runtime 混入同一个网络的正式比较。
默认 batch size：SiouxFalls 32、EMA 16、Anaheim 8。覆盖示例：

```powershell
$env:BATCH_SIZE = "4"
```

必须在该网络所有方法（包括 persistence）开始前统一设置。
`protocol.json` 会锁定数据指纹、完整配置、batch size、device 和 Hybrid OD 比例。
若 OOM 后需要改 batch，使用新的输出根目录，所有方法按新 batch 重跑，不与旧设置混合。
已完成或部分写入的单次 run 目录不会被覆盖；中断后请使用新的输出根目录，保留原现场。

## 4. 正式 45 learned + 3 persistence，顺序执行

请在尚未有正式结果的新输出目录下执行整组；不要先单独跑完某个任务再从头重复整组。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e1_matrix.ps1 `
  -PythonExe "D:\anaconda333\envs\graphgps\python.exe" -Device cuda
```

默认三个网络、三个 learned modes、seeds `42 43 44 45 46`，每个 child process 结束才开始下一个。
本地缩减为一个网络、一个 seed（仍不是正式完整结果）：

```powershell
$env:SEEDS = "42"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e1_matrix.ps1 -Networks siouxfalls -Device cpu
```

恢复五 seeds：

```powershell
Remove-Item Env:SEEDS -ErrorAction SilentlyContinue
```

Linux / Vera：先激活已有 CUDA 环境，创建 logs；单 job：

```bash
mkdir -p logs
NETWORK=anaheim MODE=old_state SEED=42 BATCH_SIZE=8 sbatch run_e1_information_set_vera_gpu.sh
```

项目根目录默认脚本所在目录；Slurm 复制脚本时应显式传项目目录：

```bash
PROJECT_ROOT="$PWD" NETWORK=anaheim MODE=old_state SEED=42 sbatch run_e1_information_set_vera_gpu.sh
```

根据所在集群要求追加账户、partition 和 GPU 型号参数，不在代码中写死他人的账户。
在已分配 GPU 的 shell 内，完整矩阵也可串行运行：

```bash
for network in siouxfalls ema anaheim; do
  NETWORK="$network" MODE=persistence bash run_e1_information_set_vera_gpu.sh || exit 1
  for mode in od_only old_state hybrid; do
    for seed in ${SEEDS:-42 43 44 45 46}; do
      NETWORK="$network" MODE="$mode" SEED="$seed" bash run_e1_information_set_vera_gpu.sh || exit 1
    done
  done
done
```

## 5. 评价定义与 test 隔离

训练只迭代 train / val，按 **validation rmse_norm** 保存一个 best checkpoint，
结束时严格 reload，之后才创建 test dataset 并进行一次正式 metrics pass。
E1 不调用 legacy final evaluator，不保存全 test predictions，不画图。
通用 custom_train 默认也关闭每 epoch test，删除了无人调用的重复 detailed evaluator。
仍使用旧输出协议的 legacy summary 函数保留，E1 不走该入口。

回归指标使用原始反标准化 prediction：micro WMAPE、RMSE、R2。
浮点导出导致的微小负 ground truth 在精度容限内归零；明显负标签报错。
预测负值不会在回归指标中被隐藏：记录所有负值比例及 `<-1e-6` 的数量。
物理指标统一显式使用 `max(pred,0)`：

- RelCon：逐图 `sum(abs(Bf-d_OD))/(sum(abs(d_OD))+eps)` 后平均。
- SUEGap：沿用 solver 中的输入校验、TNTP mask、fixed-free-flow reasonable-link mask、BPR 和 Markov loading，
  **只执行一次内层 loading**，`norm(T(f)-f)/(norm(f)+eps)`，然后逐图平均。
  不调用 outer solver 或 `verify_sue_solution`；任何 inner failure 报错。
- TSTT Error：逐图同一 BPR 的 `abs(sum(f*t(f))-sum(y*t(y)))/(sum(y*t(y))+eps)` 后平均，报告百分数。

每次只临时构建一个 CPU DiGraph，显式重排为 NetworkX 边顺序；模型参数全部取自 SUE certificates metadata。
每张真值也计算 SUEGap。任何真值 gap 超过 `max(5*generation_tolerance,1e-7)` 即停止，
该阈值只容许 float32 导出精度，不放松原始标签验收。
Runtime 使用统一 batch，5 次不计分的 warm-up，传输完成后计时并 CUDA synchronize；
不含磁盘、OD 读取、传输、loss 或物理指标。Persistence 只计对齐和投影。

闭路、容量变更、新路在同一次 test 中分别汇总，不重采样。

## 6. 汇总

```powershell
python -B scripts/summarize_e1_information_set.py --root results/e1
```

不足 48 entries 时默认报错；仅查看部分结果用 `--allow-partial`，输出明确标为 incomplete。
learned variants 为五 seeds 的 mean / sample std（ddof=1）；persistence 单值，std 留空。

- `results/e1/<network>/<mode>/seed_<seed>/summary.json`
- `results/e1/<network>/persistence/summary.json`
- 同目录 `mutation_breakdown.csv`
- `results/e1/e1_information_set_summary.csv`
- `results/e1/e1_information_set_summary.json`

`best.pt`、`history.jsonl` 和 `run_config.json` 保留以便复核；history 没有 epoch test 指标。
汇总器只读小型 JSON，验证数据/协议一致，不再加载模型或数据。
