# Paper 5 — Task-Oriented SC-BSS: Experiment Log

> 惯例：每条命令逐字记录 + 结果存档。技术设计唯一来源：
> `../docs/PAPER5_TASK_ORIENTED_PLAN.md`（含 2026-09-01 v2 修订）。

---

## 2026-09-01 — Step 0：骨架 + 基础设施（本地，CPU smoke）

**计划评估与 v2 修订**：对照 paper4 代码逐条核实原方案（详见
`../docs/PAPER5_TASK_ORIENTED_PLAN.md` v2 节）。关键修正：O1（软 SER 项
在 `variable_k_pit_loss` 内复用 PIT 指派，零额外枚举）、O2（训练集
`return_carriers` 不扰动 RNG 流——`generate_single_signal` 无条件
`return_freq=True`）、O4（相位对齐系数 detach 拟合）、O5（BER=事后 Gray
映射，仓库内无比特流）、O7（预算上调至 ~50 GPU·h）、O8（n_per_cell=100
+ 配对 bootstrap + 每调制 vs 干净源地板分析轴）。增补：config (e)
纯 λ_ser 探针 + λ_ser 扫描 Pareto 前沿。

**目录建立**：`paper5_task_oriented/` copy-not-share 自 paper4
（config/data_generator_vark/models/losses/train/evaluate/ser_comp/
aggregate_results/run*.sh），`utils.py` 符号链接到 paper1（同 paper3/4
惯例）。

**改动 1（训练集 return_carriers）**：`data_generator_vark.py`
`CommBSSVarKDataset` 新增 `return_carriers=True` 选项，返回 6 元组
`(mixture, sources, occ, k, mods, carriers)`（int64 mods 填 −1 /
float64 carriers 填 NaN，对齐测试集格式）。冒烟验证：
- configs 确定性（seed=42 两次）OK；
- 真载波 = config 载波 + 生成器内 ±5 Hz 抖动（|Δ| ≤ 5 Hz 断言通过）；
- **RNG 流守恒**：flag 开/关在相同全局 RNG 状态下 mixture/sources
  bit-identical（`np.array_equal` 断言通过）。
- 命令：`.venv_verify/bin/python data_generator_vark.py` → 全过。

**soft_demod.py（技术核心）**：torch 版软解调管道
`_soft_frontend` + `soft_ser_pairs`，与 `ser_comp._demod_pair_labels`
逐行同语义（oracle 真载波双变频 → RRC 匹配滤波 np.convolve-'same'
精确语义 → 0::sps 采样 → 单位功率归一 → detach 拟合的仅相位对齐 →
`-|s-c|²/σ²` 软对数 → CE，标签=参考硬判决 detach）。冒烟验证：
- 匹配滤波 vs `np.convolve`：rel max err 3.5e-07；
- 星座表与 `utils.CONSTELLATIONS` 逐一相等；
- Gray 标注：四种调制全部满足最近邻 1 比特差；
- **torch↔numpy 判决逐符号一致**（6 个干净源对，含 K=1/2/3）；
- identity CE ≤ 0.45（16QAM 最差，ISI 残差所致——是设计内的梯度信号），
  cross-reference CE 11–18，判别余量 ≥ 4×；
- `loss.backward()` 梯度有限且非零（|grad| sum=57.7，随机输入）。
- 命令：`.venv_verify/bin/python soft_demod.py` → 全过。

**ser_comp.py 重构**：提取 `_demod_pair_labels`（SER/BER 共享前端），
新增 `GRAY_BITS`（BPSK/QPSK 符号位、8PSK 二进制反射 Gray、16QAM 双轴
Gray）与 `compute_ser_ber_compensated`；`compute_ser_compensated` 语义
不变（薄包装）。`__main__` 基线 JSON 增加 `*_ber` 字段。

**losses.py**：`variable_k_pit_loss` 新增 `lambda_ser` / `lambda_sep` /
`ser_sigma2` / `carriers` / `mods` 参数；软 SER 项在 K 分组循环内复用
`chosen` 计算；`lambda_sep=0` 支持 config (e)（纯软 SER，无波形损失项，
PIT 指派机制不变）。冒烟验证：lambda_ser=0 时 `aux['ser_soft']==0`
（byte-identical 行为）；lambda_ser=1.0 前向/反向正常；config (e) 下
loss == ser_soft（断言通过）；缺 carriers/mods 时 AssertionError。
命令：`.venv_verify/bin/python losses.py` → 全过。

**train.py**：新增 `--lambda_ser` / `--lambda_sep` / `--ser_sigma2`；
`build_dataset` 在 lambda_ser>0 时自动开 return_carriers；训练/验证循环
支持 6 元组解包；**config (e) 专用**：lambda_sep=0 且 lambda_ser>0 时
checkpoint 选择与 LR 调度切换为 val soft-SER（min 模式）。slot-only
断言在位。

**evaluate.py**：`--ser_comp` 现输出 `ser_comp` + `ber_comp`（同源判决），
聚合与打印同步扩展；`aggregate_results.py` 的扁平化键同步。

**run.sh**：smoke（lambda_ser 开/关 + config e 各 1 epoch + ser_comp 评估）
/ `1` / full（S2 主矩阵 (a)sisdr (b)mse (c)ser (d)ser_mse × 5 seeds，
n_per_cell=100，结果入 results/s2/）/ `e`（config (e) 单 seed 探针）。
**run_ser_comp.sh**：S1 失配量化——重跑服务器上 paper4 的 15 个 mse
checkpoint（`../paper4_open_world/checkpoints/`），SER+BER，
n_per_cell=100，入 results/s1_mismatch/。

**S0 验收结果（全过）**：
- `ser_comp.py --n_per_cell 2` 基线（CPU）：SER 每调制与 paper4 的
  n_per_cell=50 参考同量级（BPSK 0.213/QPSK 0.449/8PSK 0.623/16QAM 0.786）；
  **BER 通路正确性锚点：BPSK 的 BER == SER 精确相等**（单比特调制），
  QPSK 0.249 < SER 0.449、16QAM 0.324 < 0.786（Gray 下合理）。
- `bash run.sh smoke`（CPU，PATH 指向 .venv_verify）：3 配置
  （smoke_lser0.0 / smoke_lser1.0 / smoke_pureser）各 1 epoch 全部收敛
  出 checkpoint；pureser 配置的 checkpoint args 确认
  `lambda_sep=0.0, lambda_ser=1.0`（val soft-SER 选择路径生效）；
  3 个 checkpoint 的 `evaluate.py --n_per_cell 4 --ser_comp` 全部产出
  含 `ser_comp`/`ber_comp` 的 JSON（如 lser1.0：SER 0.5113 / BER 0.2707）。
  冒烟数值本身无意义（1 epoch/64 样本），仅验证管线连通。

---

## 2026-09-01 — S1 失配量化 COMPLETE（服务器，无训练）

**命令**：`bash run_ser_comp.sh 100`（服务器 nohup，日志
`results/s1_mismatch.log`；15 个 paper4 MSE 锚定 checkpoint ×
`evaluate.py --ser_comp --n_per_cell 100` + `ser_comp.py` 基线扩展
per_snr 后重跑 n=100）。JSON 已同步回本地 `results/s1_mismatch/` +
`results/ser/baseline_ser_comp.json`。

**headline（5 seeds mean±std，n_per_cell=100）**：

| 模型 | SI-SDRi (dB) | SER | BER (Gray) |
|---|---|---|---|
| mixture 基线 | — | 0.5049 | 0.2669 |
| slot | +4.070±.098 | 0.5027±.0007 | 0.2651±.0005 |
| specialist | +4.126±.084 | 0.5018±.0004 | 0.2647±.0003 |
| recursive | +4.326±.350 | 0.4477±.0063（选择偏差，miss 37%） | 0.2310±.0039（同左） |

**零结果在 BER 下同样成立**：分离把 SER 从 0.5049 推到 0.5027（slot），
Δ≈0.002；BER 同理（0.2669→0.2651）。SI-SDRi +4 dB 撬不动比特。

**逐单元相关分析**（`make_s1_figs.py`，r 为单元内 15/10/5 次运行的
Pearson r(SI-SDRi, SER)）：
- all-15 分析在高 SNR 呈强负相关（r 最低 −0.96）——**系 recursive 的
  选择偏差伪影**（miss 37%，SER 只算在匹配对上），已在论文口径中剔除；
- slot-only / slot+specialist（诚实口径）：单元内 SI-SDRi 离散 0.16–0.47 dB，
  SER 离散 ≤0.014，r 符号随单元翻转（无稳定相关）——SI-SDR 看不见 BER。
- 每调制 vs 干净源地板：见 `results/s1_mismatch/summary.json`
  （slot 的 per-mod SER 全部贴住基线，离 0/0/0.014/0.032 地板极远——
  地板分析轴留待 S2 的任务导向模型）。

**图**：`figures/s1_sisdr_vs_ser_ber.{png,pdf}`（散点+基线）；
`figures/s1_flat_region.{png,pdf}`（平坦区：slot/specialist 带 vs
mixture 基线，recursive 单列虚线）。汇总：`results/s1_mismatch/summary.json`。

## 2026-09-01 — S2 启动（服务器，~33h）

- 四配置 GPU 冒烟（1 epoch, 64 samples）全部正常，ser_soft 初值 ~3.1–3.2，
  无发散；config (e) 的 val soft-SER 选择路径工作正常。
- 命令：`bash run.sh full && bash run.sh e`（nohup，日志
  `run_s2_full.log`）——(a)sisdr (b)mse (c)ser (d)ser_mse × seeds 42–46
  + (e) pureser 单 seed 探针；每个 run 后接 `evaluate.py --ser_comp
  --n_per_cell 100 --out_dir results/s2`。
- **故障与修复（2026-09-01 21:5x）**：首次启动立即失败
  （`ModuleNotFoundError: numpy`）——`run.sh` 用 `python3` 但未像
  `run_ser_comp.sh` 那样导出服务器 venv PATH。已在 `run.sh` 顶部加
  `export PATH=/data/experiment/venv_bss/bin:$PATH` 并重启；epoch 1 内
  SI-SDR 正常爬升（GPU 98%）。教训：paper4 的 run.sh 注释"Runs locally"
  在服务器场景必须显式 venv。

## 2026-09-02 — S2 主矩阵 COMPLETE + 中期分析

21 个 eval JSON 已回收到 `results/s2/`；分析脚本 `analyze_s2.py`（配对
bootstrap 95% CI，seed 级 n=5 + 单元级 n=35）；汇总
`results/s2/summary_s2.json`；图 `figures/s2_pareto.{png,pdf}`。

**Headline（5 seeds mean±std）**：

| 配置 | SI-SDRi (dB) | SER | BER | count_acc(occ) | halluc |
|---|---|---|---|---|---|
| (a) sisdr | +4.230±.024 | 0.5022±.0002 | 0.2646±.0001 | 0.649±.018 | 0.155 |
| (b) mse | +4.070±.098 | 0.5027±.0007 | 0.2651±.0005 | 0.658±.023 | 0.149 |
| (c) ser | +3.909±.023 | 0.5031±.0017 | 0.2638±.0009 | 0.850±.088 | 0.060 |
| (d) ser_mse | +3.721±.098 | 0.5034±.0013 | 0.2647±.0011 | 0.857±.032 | 0.049 |
| (e) pureser (n=1) | −9.418 | 0.5159 | 0.2713 | 0.913 | 0.034 |
| mixture 基线 | — | 0.5049 | 0.2669 | — | — |

**读数（诚实版，2026-09-03 置换检验后修订）**：
1. **SER/BER：均无统计上显著的差异**。补做精确配对置换检验
   （sign-flip，2^5=32 全枚举）后：ser vs sisdr BER Δ=+0.00084
   perm_p=0.1875；ser vs mse BER Δ=+0.00133 perm_p=0.1250；
   ser_mse vs sisdr SER Δ=−0.00122 perm_p=0.1875。注意 n=5 时精确
   置换检验的功效上限即 p=0.0625（全符号一致才达到），因此此前用
   bootstrap CI 报告的"BER 显著改善"**予以撤回**——小样本 bootstrap
   夸大了显著性。正确表述：软解调项对 SER/BER 的影响在 |Δ|≤0.13pp
   以内且方向不稳定，**任务导向训练对比特的效应为零或淹没在噪声中**。
2. **意外正效应**：软解调项把 occupancy 路由的 K=2 计数从 0.001 修到
   ~0.77（occ overall 0.65→0.85），halluc 率 0.15→0.05。可作为
   "任务导向监督正则化占用结构"的辅助卖点。（注：paper4 headline
   计数 0.93 走的是 count_head 路由；occupancy 路由在 (a)/(b) 下
   K=2 基本失效——S2 四配置同一评估代码，横向对比成立；但该效应
   方差较大（occ_acc std 0.088），表述限定为涌现式 occupancy 路由。）
3. **(e) pureser 负面结果**：λ_sep=0 时 ser_soft 正常下降
   （2.24→1.81，说明软项单独可优化），但 SI-SDRi −9.4 dB、SER 0.516
   **差于基线**——模型找到了最小化软 CE 却不分离波形的离流形解
   （对软接收机的 reward hacking）。结论："波形目标必要但不充分"。
   注意：代理-真相一致性只在近净流形上成立（S0 验证的是干净/恒等
   输入），论文限制节必须写明。
4. **(c) 的 ser_soft 轨迹几乎平**（2.95→2.84）——λ=1.0 时软项相对
   SI-SDR 梯度太弱。
5. **隐式均衡假设未发生**：ser 模型 per-mod SER（0.234/0.411/0.619/
   0.742）离干净源地板（0/0/0.014/0.032）很远——任务导向训练未表现
   出对多径 ISI 的隐式均衡（O8 分析轴的否定答案，一句话写入论文）。

**消融已启动**（`run_ablations.sh`，nohup `run_ablations.log`）：
λ_ser ∈ {0.3, 3.0}、σ² ∈ {0.05, 0.2}（单 seed 42，~6h）。
λ_ser=3.0 的剂量-响应是判定"软项是否有救"的关键。

## 2026-09-03 — 消融结果 + S3 实现与启动

**消融结果**（单 seed 42；对照 main s42：sisdr +4.258/0.5020/0.2644，
ser +3.904/0.5018/0.2631）：

| run | λ_ser | σ² | SI-SDRi | SER | BER | occ_acc | halluc |
|---|---|---|---|---|---|---|---|
| ser_l0.3 | 0.3 | 0.1 | +4.188 | 0.5029 | 0.2643 | 0.768 | 0.105 |
| ser_l3.0 | 3.0 | 0.1 | +2.668 | 0.5046 | 0.2647 | 0.932 | 0.023 |
| ser_s0.05 | 1.0 | 0.05 | +3.156 | 0.5054 | 0.2655 | 0.922 | 0.028 |
| ser_s0.2 | 1.0 | 0.2 | +4.130 | 0.5033 | 0.2644 | 0.831 | 0.077 |

**剂量-响应判定**：λ_ser 单调换的是"SI-SDRi↓ + occ_acc↑ + halluc↓"，
**SER/BER 全程无剂量-响应**（0.502–0.505 / 0.263–0.265）。直接软解调
损失此路不通（对 SER/BER），论文主线按 §8 降级路线转向 S3 联合检测；
计数正则化效应保留为辅助贡献。

**S3 实现（JointLLRHead）**：
- `models.py`：`JointLLRHead`（槽位共享；输入 = 槽位掩蔽特征 ⊕ 未门控
  bottleneck 拼接 [B,K,2H,T] → 1×1 复卷积 → sps 窗均值降到符号率 →
  2×复卷积块 → 16 维符号 logits；+30.3K 参数，总计 284.4K）；
  `SlotSepNet(use_joint_head=...)` 返回 4 元组。冒烟：形状/反传 OK。
- `soft_demod.py::ref_hard_labels`（torch 批量参考硬标签，与
  `_soft_frontend` 参考侧同语义）；`ser_comp.py::ref_labels_only`
  （numpy 版，评估用）。
- `losses.py`：`variable_k_pit_loss` 新增 `lambda_llr`/`sym_logits`
  —— 在 PIT 指派槽位上对头部 logits 做星座掩码 CE。冒烟：llr 项
  有限、可反传、缺参守卫 OK。
- `train.py`：`--lambda_llr`（slot 限定），lambda_llr>0 自动开
  return_carriers、自动建联合头；llr 入训练日志。
- `evaluate.py`：ckpt 自动检测 lambda_llr>0 → 建联合头 + 测试集
  return_carriers；`_match_and_measure` 新增 `est_syms`——对
  Hungarian 匹配对输出 `ser_joint`/`ber_joint`（头部掩码 argmax vs
  参考硬标签，SER/BER 同源）；`_sep_block`/打印同步（per_snr/per_k
  自动继承）。
- **冒烟修复**：`ref_labels_only` 入参需 torch 张量（评估侧传的是
  numpy），已修。冒烟评估（1 epoch）：波形式 SER 0.477/BER 0.262 正常；
  联合头 SER 0.723/BER 0.501 ≈ 随机（BPSK 0.504、16QAM 0.919≈15/16，
  Gray 随机比特 BER=0.500 锚点精确），管线正确。

**S3 启动**：`bash run_s3.sh`（服务器 nohup，日志 `run_s3.log`）——
定 K=2（k_min=k_max=2，k_slots=4 保全测试网格），λ_llr=1.0 +
λ_mse=1.0，5 seeds，每 run 接 `--ser_comp --n_per_cell 100`（波形式 +
联合头双路由）。预计 ~8h。

## 2026-09-03 — S3 v1 结果：联合头失败 + 根因定位 + v2 头

**v1 结果**（5 seeds，`results/s3/summary_s3.json`，分析脚本
`analyze_s3.py`）：K=2 单元 —— 联合头路由 SER 0.7324±.0021 /
BER 0.4668±.0015，**远差于**基线（0.5612/0.2872）和波形式路由
（0.5475/0.2777）；配对 bootstrap CI 全部远离 0（联合头差 0.17–0.19）。
波形路由不受损（SI-SDRi(K=2) +3.41 dB，对齐 P4 fixedK2 的 3.61）。

**根因（v1，措辞修正 2026-09-03）**：v1 头的"帧率→符号率"用窗均值
（每 sps=16 样本取均值），载波 2000 Hz @ fs 16000 在每 16 样本窗内
恰好转满 2 整圈——boxcar 均值精确为 0。该算术对**波形通道**严格成立；
v1 头输入为学习特征，"特征同样贫化"是推断而非直接证实。训练轨迹
（llr 2.2→1.68 平台期，16QAM 联合 SER≈0.92≈随机）与之一致。

**v2 修复**（`models.py::JointLLRHead`）：
1. 窗均值换成**可学习跨步复卷积**（k=2·sps=32, stride=sps, pad=sps/2）
   ——头可以自学下变频+匹配滤波；
2. 输入追加**槽位波形本身**（`slots` 通道）——保证头能接触到载波相干
   信号（复用与波形式接收机相同的物理信息源）。
参数 96.1K（总计 350.2K）。冒烟（本地）：形状/反传 OK。

**待办**：v2 头 5 seeds 重跑（run_s3.sh 不变），与 v1 同口径对比。

## 2026-09-03 — S3 v2 结果：仍失败 → 更深的根因 → S3 终止（按计划降级路线）

**v2 结果**（5 seeds，`results/s3/summary_s3.json`，v1 归档
`results/s3_v1_meanpool/`）：K=2 单元联合头 SER 0.7297±.0018 /
BER 0.4639±.0011 —— 与 v1（0.7324/0.4668）**统计上无差别**；
波形路由依然完好（SER 0.5437/BER 0.2758，SI-SDRi(K=2) +3.46 dB）。
训练轨迹：llr 1.70→1.67，与 v1 同一平台期。

**v2 失败排除"窗均值湮灭载波"后的更深根因 → 探针实证（2026-09-03）**：
即使载波可学消除，每源还有**任意初始相位 + 残差频偏**；评估接收机
（ser_comp）用 oracle 真载波下变频 + 在全部 256 个符号上拟合相位来
消除它们，而符号率卷积头的感受野只有 ±4 符号。**受控探针
`probe_sync_head.py`（K=1、无分离无 PIT 的 A/B 实验，服务器
`probe_sync_head.log`）证实了这一因果链**：
- 变体 P（通带输入，需盲同步）：val SER 0.680 ≈ 随机（loss 1.57 平台
  期；BPSK 0.384 略有可学性，QPSK+ 全部随机）；
- 变体 S（oracle 下变频+突发级相位对齐，同架构同数据）：val SER 0.150
  且仍在下降（BPSK 0.028 / QPSK 0.087 / 8PSK 0.249 / 16QAM 0.236）。
- 判定规则预设"S 学成且 P 失败 ⇒ 盲同步是瓶颈"——**成立**。S3 v2 的
  失败因果归于突发级盲载波/相位恢复，而非头部容量或训练管线。
  要让联合头工作必须引入突发级相位估计通路（或对头部也给 oracle
  同步）——超出 S3"冲刺"预算，按计划 §8 终止并作为有实证根因的
  负面结果入稿。

**S3 终止决定**（计划 §8 预设路线）：联合 LLR 头两版均告失败，作为
**有根因诊断的负面结果**写入论文（v1：窗均值湮灭载波——架构陷阱；
v2：盲相位恢复超出感受野——联合检测的神经化需要显式同步机制）。
论文最终叙事定型：
1. S1：SI-SDR↔SER/BER 失配量化（含 recursive 选择偏差伪影分析）；
2. S2：直接软解调训练对比特无显著效应（置换检验 perm_p≥0.125，
   |Δ|≤0.13pp），但显著改善计数/占用结构（occ 0.65→0.85，幻觉
   0.15→0.05）；pureser 确立"波形目标必要不充分"（代理可被
   reward-hack）；
3. S3：联合软解调的朴素神经化失败，根因两级诊断。
卖点从"联合检测更优"转为**"SIR≈0 同频混叠下，波形指标、任务损失、
朴素联合检测都撬不动比特——问题需要显式同步/均衡机制"**的失配研究。

**写作素材齐备**：图 s1_sisdr_vs_ser_ber / s1_flat_region / s2_pareto；
表 S2 主矩阵 + 消融 + S3 两版；全部命令与数字在本日志。


---

## 2026-09-08 — 写作素材补齐：S3 图 + 三张 LaTeX 表

**命令**：`../.venv_verify/bin/python make_s3_figs.py`、`../.venv_verify/bin/python make_tables.py`（本地，纯读取已有 JSON/log，无新实验）。

**新增脚本**：
- `make_s3_figs.py`：读 `results/s3/summary_s3.json`（v2）、
  `results/s3_v1_meanpool/summary_s3.json`（v1）、
  `results/probe_sync_head.log`（正则解析 P/S 学习曲线）。
- `make_tables.py`：读 `results/s2/summary_s2.json` +
  `results/s2_abl/*.json` + `results/s2/` 单 seed 对照 JSON。

**新增产物**：
- 图 `figures/s3_joint_vs_waveform.{png,pdf}`：K=2 单元 SER/BER 双 panel
  柱状（mixture 0.5612/0.2872；S3 波形路由 0.5437±.0083/0.2758±.0056；
  联合头 v1 0.7324/0.4668、v2 0.7297/0.4639 ≈ chance 0.766/0.500）。
- 图 `figures/s3_probe_sync.{png,pdf}`：探针 P（盲同步 0.697→0.681 平台）
  vs S（oracle 同步 0.312→0.150）学习曲线 + ep30 per-mod 柱状
  （BPSK 0.38 vs 0.03，16QAM 0.81 vs 0.24）。
- 表 `paper5/tables/tab_s2_main.tex`（主矩阵 5 配置 + 基线，表注含三个
  精确置换检验 Δ/p 与 n=5 功效上限说明）、`tab_s2_ablation.tex`
  （λ_ser/σ² 消融，读数与本日志 2026-09-03 消融表逐项一致）、
  `tab_s3.tex`（K=2 路由对比 + paired bootstrap CI + 两级根因）。
- 两张 S3 图的 pdf 已复制到 `paper5/figures/`（共 5 张图就位）。

**写稿注意（数据层面的张力，如实处理）**：
1. bootstrap CI 与置换 p 存在冲突（ser vs sisdr BER 的 CI95 不含 0 但
   perm_p=0.1875）——表注统一按置换检验口径"无统计显著效应"，审稿人
   可能追问 CI，正文中需一句话解释 n=5 功效上限。
2. S2 表的基线 SER 0.5049 是 K∈{1,2,3} 混合单元，S3 表的 0.5612 是
   K=2 单元，不可跨表直接比较（两表注均已标明）。

---

## 2026-09-08 — 外部评审意见核实与处置

收到一份对 paper5 的深度技术评审（覆盖 soft_demod/ser_comp/losses/models/
data_generator_vark/train/evaluate 七文件逐行比对）。事实性断言全部核实
如下，处置按"低风险代码修复立即做、叙事层留到写稿"原则执行。

**已立即修复（行为不变，仅防护/注释）**：
- `paper5/build.sh:2` 注释残留 "paper3" → 改为 paper5（确认 bug）。
- `soft_demod.py` 两处 `sps = T // n_symbols` 前加
  `assert T % n_symbols == 0`（`_soft_frontend` + `ref_hard_labels`）。
- `models.py::JointLLRHead.forward` 加 `assert T % sps == 0`。
  注：评审称 sps=16 "硬编码"——实为带默认值的构造参数（SlotSepNet 透传），
  评审此点措辞偏重；仍补 assert 防配置漂移。
- 验收：`soft_demod.py` / `models.py` 的 `__main__` 冒烟全部通过
  （torch↔numpy 逐符号一致、rel err 3.5e-07、反传正常）。

**核实属实、留到写稿处理（叙事/口径层，不改数据）**：
1. 机制性论证（软 SER 经 oracle 同步+相位对齐后与 SI-SDR 的可行集在
   SIR≈0 下高度重叠——零效应有机制必然性）：评审判断与本日志
   2026-09-03 的证据链一致（无剂量-响应、ser_soft 平台期、pureser
   reward-hack），采纳为正文核心论证；表述上注意软 SER 并不等同于
   SI-SDR（作用于匹配滤波后符号网格、尺度/相位不变），应写为
   "两者看见同一瓶颈（幅度级干扰泄漏）"而非"完全退化"。
2. (c) vs (b) 对比不对称：属实——(c) 是"软 SER **替代** MSE"，
   加法公平对照是 (d) vs (b)（SER 0.5034 vs 0.5027、BER 0.2647 vs
   0.2651，均无显著）。正文必须点明，headline 对比改用 (d) vs (b)。
3. oracle 同步假设（S2 训练损失与 ser_comp 评估真相两端都吃真载波）：
   属实，正文显著声明；ser_comp 系 paper4 沿袭的"补偿接收机"上限口径。
4. n=5 置换检验功效上限 p=0.0625：属实（2/2^5）。正文措辞改为
   "|Δ|≤0.13pp 且方向不稳定"，而非"无显著差异"。可选增强：对单元级
   配对（n=35 cells，analyze_s2 已算 cell 级 Δ/CI）补一个置换检验，
   功效远高于 seed 级——写稿时决定是否补跑。
5. std 为 population std（np.std 默认 ddof=0，analyze_s2 与
   make_tables 一致）：属实。决议：不改历史数据/表（避免与日志脱节），
   正文声明 "std across seeds (population, ddof=0)"。
6. `summary_s2.json` 的 count_acc 无路由标签：属实（取 occupancy 路由，
   `analyze_s2.py:94`）。决议：不改归档 JSON；表格 caption 已写明
   occupancy 路由（tab_s2_main/tab_s2_ablation），写稿时注意不与
   count_head 路由（~0.91）混用。
7. logits 用对齐后 est、labels 用未对齐 ref 的隐含约定：属实且自洽
   （与 ser_comp 同语义），正文一句话交代。

---

## 2026-09-08 — 单元级（n=35）置换检验补跑：零效应结论细化

**动机**：seed 级精确置换检验（n=5）功效上限 p=0.0625，无法区分
"效应为零"与"功效不足"。评审建议正视此点；补跑单元级检验。

**命令**：`../.venv_verify/bin/python analyze_s2_cells.py`（本地，
新脚本；7 SNR 单元 × 5 seeds = 35 对配对差，10^6 Monte-Carlo
sign-flip，rng seed 0，p=(1+#)/ (1+N)）。输出
`results/s2/summary_s2_cells_perm.json`。

**结果**（Δ 正 = 前者更优；8 个对比 Bonferroni α=0.00625）：

| 对比 | 指标 | Δ | mc_p | 判定 |
|---|---|---|---|---|
| ser vs sisdr | BER | +0.00102 | 0.0004 | **显著**（过 Bonferroni） |
| ser vs mse | BER | +0.00169 | 0.0001 | **显著**（过 Bonferroni） |
| ser vs sisdr | SER | −0.00060 | 0.171 | n.s. |
| ser vs mse | SER | +0.00017 | 0.793 | n.s. |
| ser_mse vs mse（公平加法对照） | SER/BER | −0.00097 / +0.00027 | 0.38 / 0.61 | n.s. |
| ser_mse vs sisdr | SER/BER | −0.00173 / −0.00041 | 0.064 / 0.649 | n.s. |

**稳健性核查**（同日）：
- 效应集中在**高 SNR 单元**（10/15/20 dB：+0.2~0.5pp，逐 seed 一致为正），
  低 SNR ≈0——与机制解释自洽（高 SNR 残差是网格抖动、接收机受限，
  软 SER 可救；低 SNR 是幅度级干扰泄漏，无救）。
- 高 SNR 单元均值回到 seed 级严格检验：4/5 seeds 同号（seed 46 ≈0），
  精确 perm_p=0.125（未达 0.0625 地板但方向一致）——结论不依赖
  单元独立性假设的方向性佐证。
- 注意事项：单元级 sign-flip 假设同 seed 内 7 单元差分对称可交换；
  已用上述 seed 级复查兜底，论文中 cell-level p 需一句话交代该假设。

**结论修订（写稿口径）**：从"无显著效应"细化为——
**(c) 的软 SER 项在 BER 上有统计真实但实用可忽略的提升（≤0.17pp，
仅高 SNR）；公平加法对照 (d) vs (b) 与全部 SER 对比均为零**。
干预受限单元（论文的动机场景）依然纹丝不动。

`make_tables.py` 的 `tab_s2_main.tex` caption 已同步重写（含 population
std 声明、(c) 替代 MSE 的口径说明、cell-level 结果与高 SNR 集中性），
三表已重新生成。
`make_tables.py` 的 `tab_s2_main.tex` caption 已同步重写（含 population
std 声明、(c) 替代 MSE 的口径说明、cell-level 结果与高 SNR 集中性），
三表已重新生成。

---

## 2026-09-08 — `paper5/main.tex` 初稿完成（目标期刊 AEÜ）

**产出**：
- `paper5/main.tex`（全新，~860 行，elsarticle review 12pt，风格与
  文献格式照 paper3；内联 thebibliography 24 条，含
  `nong2026openworld` 自引 companion"under review"条目）。
  标题：*Task-Oriented Single-Channel Blind Source Separation:
  Bit-Level Gains Are Interference-Limited, Not Loss-Limited*。
  结构：Intro（5 条 findings）→ Related（SC-BSS / 波形指标批评 /
  任务导向通信 / 学习检测）→ §3 Benchmark + 补偿接收机 + Gray BER
  （oracle 假设显著声明、identity=0、干净地板 0/0/0.014/0.032）
  → §4 Method（SlotSepNet 简述 + 软解调损失 Eq.(2)(3) + 配置 (a)-(e)
  定义 + JointLLRHead v1/v2）→ §5 Experiments（5.1 统计协议：
  population std、seed 级精确置换功效上限 0.0625、cell 级 n=35
  MC 检验 + Bonferroni 0.05/8、occupancy 路由声明；5.2 S1 失配 +
  recursive 选择偏差伪影；5.3 S2 主矩阵三读数；5.4 消融无剂量-响应
  + 无隐式均衡 + pureser reward hacking；5.5 S3 两版失败 + 探针诊断）
  → §6 Discussion（机制论证：SIR≈0 下两目标看见同一瓶颈=幅度级干扰
  泄漏；高 SNR 接收机受限才轮到软 SER；(c) 替代 MSE 的口径点明、
  公平对照 (d) vs (b) 为零；计数正则化解释；指向显式同步/均衡）
  → §7 Limitations（oracle 同步两端、合成数据、K≤3、ISI 标签噪声、
  cell 级可交换性假设、单 seed 探针标注）→ §8 Conclusion。
- `paper5/abstract.txt` / `highlights.txt`（5 条均 ≤85 字符）/
  `keywords.txt`（6 个，与 paper3 同格式）。

**图表接线**：5 图全部 `\includegraphics` 自 `paper5/figures/*.pdf`；
3 表 `\input{tables/tab_*}`（表注即数据真相，正文数字与之一致）。

**生成器同步改动**（`make_tables.py`，已重新生成三表）：
- `tab_s2_main` 行标签去掉 λ 注释（避免 84pt→19pt→0 的 overfull 迭代），
  字号 footnotesize→scriptsize、tabcolsep 2pt；caption 增补
  "Configs (a)–(e) as defined in Section 4.3"。
- 三表浮动体 `[t]`→`[H]`：elsarticle review 格式下 `[t]` 表会被
  冲到 References 之后的浮动堆（已在 PDF 中复现并修复），paper3
  的表本来就用 `[H]`。

**编译**：`cd paper5 && bash build.sh`（XeLaTeX×2）30 页，pass 2 无
undefined ref/citation、无 overfull；仅存 1 个 URL underfull
（data availability 的 GitHub 链接，与 paper3 同款，可接受）。
标题页/图 1/图 2/表 1 已渲染目检正常。

**未做（留待投稿前）**：cover_letter.tex（build.sh 已支持
`bash build.sh cover`，模板可从 paper2/3 移植）；文献 DOI 复核
（新引 Strinati2021/Gündüz2023/Samuel2019/Farsad2018/Hershey2016
五条建议投稿前逐条过一遍 CrossRef）；AGENTS.md 的 paper5 条目
状态更新（S0→manuscript drafted）。

---

## 2026-09-08 — 第二轮外部评审：per-K 符号反转发现 + 五处修订（全部核实属实）

收到第二轮深度评审（通读 main.tex + 三表 + 日志 + results/s2 全部
eval JSON 重算）。**数据类断言全部核实属实**，论文按此修订；措辞类
建议逐条采纳/对冲。本轮无新训练，全部为既有 JSON 的重聚合。

**核实 1（最重要）：per-K 分解揭示 K=2/K=3 方向反转，pooled 相互抵消**。
每个 eval JSON 本有 `separation.per_k.{ser_comp,ber_comp}`，此前从未聚合。
新脚本 `analyze_s2_perk.py`（命令：`../.venv_verify/bin/python
analyze_s2_perk.py`，本地；4 对比 × SER/BER × K∈{1,2,3}，seed 级
n=5 精确 sign-flip 2^5 全枚举；Δ 正=前者更优），输出
`results/s2/summary_s2_perk.json`。结果复现评审全部数字，并补全
四个对比的完整图景：

| 对比 | 指标 | K=1 | K=2 | K=3 |
|---|---|---|---|---|
| (c)vs(a) | BER | −0.01pp (3/5) | **+0.35pp (5/5, p=0.0625)** | **−0.15pp (0/5, p=0.0625)** |
| (c)vs(a) | SER | −0.03 (2/5) | +0.53 (4/5) | −0.72 (0/5) |
| (c)vs(b) | BER | −0.19 (2/5) | **+0.40 (5/5)** | −0.14 (0/5) |
| (d)vs(b) | BER | +0.26 (4/5) | +0.11 (5/5) | −0.10 (1/5) |
| (d)vs(b) | SER | +0.53 (4/5) | +0.32 (4/5) | −0.60 (0/5) |

**K=2 BER 在全部 4 个对比中 5/5 为正；K=3 SER 在全部 4 个对比中
0/5 为正（p=0.0625 功效地板）**。1:2:3 对数加权后抵消为 pooled
≤0.17pp。论文结论从"干扰受限单元纹丝不动"修订为"效应符号由干扰
源数决定：K=2 一致获益、K=3 一致受损、pooled 可忽略"——这比原结论
更强（直接操纵干扰维度，而非借 SNR 代理），新表
`tables/tab_s2_perk.tex`（make_tables.py 新函数）入 §5.3 读数 2，
机制论证（§6）改写为 K 轴：K=1 无干扰无所得；K=2 单干扰残差有结构、
网格梯度可推边界符号；K=3 双干扰残差填满网格、拉向错误星座
（标注为与符号图案一致的假设，符号图案本身是稳健观察）。

**核实 2：headline 归属错误，属实**。0.17pp 出自 (c)vs(b)（替换 MSE
锚），公平加法对照 (d)vs(b) 实为 **0.027pp**（cell Δ=+0.00027,
p=0.61）。摘要/Intro/§5.3 读数 1 已全部改为"pooled ≤0.17pp（替换
配置 (c)）；公平加法 ≤0.03pp"。

**核实 3：§6 技术错误，属实**。SI-SDR 对时移敏感（非不变），且
ser_comp 根本不做定时搜索（§3.2 自述）。原文"移除 scale/phase/timing
这些 SI-SDR 已不变的量"已删，改为"软接收机只移除 scale 与 phase 两个
自由度——与 SI-SDR 不变的集合相同；定时/ISI 误差不属于任一目标的
可达集（§5.4 已证无隐式均衡）"。

**核实 4：§5.2 "无任何稳定单调通道"过度断言，属实**。slot-only
逐单元 r（summary_s2.json `per_cell_correlation`）：−10:+0.11 /
−5:−0.65 / 0:+0.57 / 5:−0.87 / 10:−0.69 / 15:**−0.98** / 20:−0.09
——7 单元中 4 个强负相关（"分离越好 SER 越低"的正确方向）。日志此前
将强负相关归因于 recursive 选择偏差，但 slot-only 无 recursive。
正文改为"通道存在但实用上惰性：单元内 0.16–0.47 dB SI-SDRi 仅换
≤0.014 SER"。

**核实 5：(d)vs(a) SER 边际负面被遗漏，属实**。cell-level
Δ=−0.00173，bootstrap CI95 [−0.00377, −0.00005] 不跨 0，mc_p=0.064
（置换口径 n.s.）。§5.3 读数 1 已补披露句（含 n=5 下 bootstrap CI
反保守、置换为primary 的口径交代，及该负尾即 K=3 效应的指向）。

**核实 6（被浪费的证据，已采用）**：per-SNR SI-SDRi +8.42dB(−10dB)
单调降至 +1.30dB(20dB)（s1_mismatch slot 5 seeds 均值复核）——
"−10dB 单元 +8.4dB 换不来任何 SER 改善"写入摘要/Intro/§5.2/结论；
mixture baseline per_k SER 0.146(K=1)/0.561(K=2)/0.587(K=3)
（results/ser/baseline_ser_comp.json）——"单个干扰源造成 3.8× 跳变、
第二个几乎不再增加"作为 interference-limited 的单点证据写入 §5.2。

**措辞对冲（按日志 2026-09-03 原记载执行）**：v1 "annihilated
arithmetically" 改为"对通带波形算术上精确为零；头部输入为学习特征，
载波贫化系推断（架构 + 16QAM≈随机平台期）而非直接证实"；探针归因
软化为"指向盲同步为 binding constraint；探针是 K=1 而失败在 K=2，
不排除次级容量/优化因素"。

**Related Work 补两条线**：PCMA/同频干扰抵消（Dankberg 1998 AIAA
ICSSC 已网络核实存在；Andrews 2005 IEEE Wireless Commun. 12(2):19-29）
——§2.4 新增"显式同步+均衡是该社区二十年成熟方案"定位；语音侧
task-loss 正面例（MetricGAN, Lo et al. ICML 2019；Ochiai et al.
ICML 2017 E2E ASR）——§2.3 新增"语音上有效、同频通信上受干扰结构
约束"的对比。§2.1 "none evaluates" 软化为"bit-level read-outs are
rare ... no prior work tests under symmetric co-frequency mixtures"。

**计数卖点自保**：§5.3 读数 3 增加 caveat+defence——count_head 路由
~0.91 仍更优，任务项修的是 occupancy **门控**（决定哪些波形送给下游；
halluc 输出是实际波形而非只是计数错），0.15→0.05 是比特流相关的改善。

**口径修正**：§5.1 统计协议新增 Δ 符号约定（"X vs Y"：Δ=metric(Y)
−metric(X)，正=前者更优）。

**处置为"不适用"的两条**：①highlights 入 main.tex——Elsevier 惯例为
分离文件（paper2/3 的 main.tex 均无 highlights 环境），无需改；
②自引破盲——AEÜ 官方 Guide for Authors 明确 single anonymized
review（已网络核实），自引安全。

**产物**：main.tex 35 页（+5），`bash build.sh` 两遍无 undefined、
无 overfull；新表 tab_s2_perk.tex 已目检渲染正常；abstract.txt /
highlights.txt 同步更新。新脚本 analyze_s2_perk.py 与产物
summary_s2_perk.json 归档。

**遗留（评审的策略性建议，未动）**：cover letter 定位（measurement/
negative-result paper）与备选刊（IEEE TCCN / WCL correspondence）——
投稿前由作者定夺；5 条新引文献中 Andrews 2005 未附 DOI（不确定、
宁缺勿造），投稿前建议统一过 CrossRef。

## 2026-09-08 — 初稿第三轮外部评审处置（11 条：P0×2 / P1×3 / P2×5+1）

评审范围：main.tex（1137 行）+ abstract/highlights/keywords + 四张表，
逐条与日志/run.sh/analyze 脚本/summary JSON 交叉核对。结论：前两轮修订
已全部落地，本轮 11 条均为表述层。逐条核实与处置如下。

**P0-1 §4.4 "16-way logits over the union constellation"——属实，已改，
但评审建议的替换措辞本身不准确。** 并集是 2+4+8+16=30 点，原文自相矛盾
成立。但实现不是评审所说的"16QAM 网格作共享标签空间"：labels 来自
ser_comp/soft_demod 的 argmin（对**该样本自己调制**的星座取索引 0..M-1），
四种调制星座几何上互不嵌套（BPSK ±1、QPSK ±(1±j)/√2、8PSK 单位圆、
16QAM {±1,±3}/√10——同索引 j 在不同调制下是不同几何点）。真实语义是
**容量 M_max=16 的共享索引空间 + per-modulation 掩码激活前 M 项**
（CONST_MASK[m,:M]=True，soft_demod.py:75；losses.py:245-249）。
正文改为 "shared label space of capacity M_max=16, label j = j-th point
of the sample's own constellation, per-modulation mask activates the first
M∈{2,4,8,16} entries"；models.py JointLLRHead docstring 同步修正。

**P0-2 摘要孤立引用 p=0.0625——属实，已改。** 摘要 K=2 处改为 "5/5 seeds
positive in all four contrasts"（删孤立 p 值，正文 §5.3 保留 p 与功效上限
上下文），K=3 对称补 "0/5 seeds positive in every contrast"。abstract.txt
同步。

**P1-3 tab_s3 "S2-ser waveform route" 孤儿行——属实，已补解释。** 核实该行
= summary_s3.json 的 s2_routes_k2['ser']（S2 (c) 模型在 K=2 单元的波形路由，
5 seeds 均值 0.54749/0.27770，与表一致；s2_routes 还含 sisdr/mse 两行未入表）。
作用=跨研究锚点，§5.5 补一句：该行与 S3 自身波形路由（0.5437/0.2758）
差在一个 std 内，证明头失败时骨干处于预期工作点。

**P1-4 (b) 与 S1 slot 数字逐位相同——属实且比评审说的更强。** results/s2/
与 results/s1_mismatch/ 的 slot_mse 五 seed eval JSON **整文件字节相同**
（diff 验证；mtime 相差 ~5.5h，分别为两次独立运行产出）。S1 评估的是
paper4 的 checkpoint（run_ser_comp.sh:26 指向 ../paper4_open_world/
checkpoints/），S2 (b) 是 paper5 自己重训的（run.sh CONFIGS 含 "mse 1.0 0.0"）。
字节相同 ⟹ 两次训练比特级确定：数据 RNG 流按设计相同（data_generator_vark
为 paper4 拷贝 + return_carriers，RNG-stream identical），torch.manual_seed
固定，服务器上训练全程确定性。这是**完美的配方复现证据**（非复用、非错误）。
§5.1 已加注："coincide by construction, not by reuse"。

**P1-5 §5.3 同一对比两个 Δ——属实，已标注。** seed-level Δ=−0.0012/p=0.1875
与末段 cell-level Δ=−0.00173/p=0.064 是两级检验，末句已补 "cell-level"。

**P2 处置**：
- highlights.txt 的 ->/<= 改为英文单词（to / at most）——已改。
- tab_s2_main caption "0.05/8=0.006" → 0.00625（make_tables.py:91 已改，
  表格重生成）。
- §4.4 "kernel 2×16, stride 16" → "1-D kernel spanning two symbols,
  i.e. 32 samples, stride 16"——已改。
- §3.2 "naive proxy SER≈0.68" **无出处——属实，且实测值不是 0.68**。
  日志与 paper1 日志均无该数字来源。新写 check_naive_proxy.py 实测
  （253 个 16QAM 真源，test seed 99999，oracle 路径作真值）：
  oracle 0.0000 / naive（标称载波、无定时无尺度校正）**0.9203** /
  naive+单位功率归一 **0.7734**。正文改为 0.92（0.77 with unit-power
  normalisation）并引用脚本。注意 paper1 的 compute_ser_from_signal
  对 est=ref 恒给 0（同路径自比），不可能是 0.68 的来源。
- baseline K=2(0.561) vs K=3(0.587) 饱和——§5.2 原有 "the first
  interferer does nearly all of it"，已补半句点明 "baseline already sits
  near the saturated separation-failure level at K=2, so K=3 is not
  proportionally harder"。

**产物**：main.tex 36 页（+1），build.sh 两遍编译干净、无 overfull；
表格重生成；abstract.txt/highlights.txt 同步。新脚本 check_naive_proxy.py
归档（可复跑验证 §3.2 数字）。

## 2026-09-08 — 投稿准备：文献 CrossRef 核对 + cover letter

**文献核对（24 条全过 CrossRef/PMLR，发现 4 处实错 + 2 处补 DOI）**：
- hershey2016deep：DOI 误为 10.1109/ICASSP.2016.7471618（该 DOI 实为
  ICASSP 2016 **front matter**），正确为 **10.1109/ICASSP.2016.7471631**
  （pp.31-35 吻合）——已修。
- samuel2019learning：DOI 末位错 2，正确 **10.1109/TSP.2019.2899805**
  （TSP 67(10):2554-2564 吻合）——已修。
- farsad2018neural：DOI 错，正确 **10.1109/TSP.2018.2868322**
  （TSP 66(21):5663-5678 吻合）——已修。
- lo2019metricgan：**作者列表错配**（误用了 MOSNet 的作者
  C.-C. Lo et al.；MetricGAN 实为 S.-W. Fu, C.-F. Liao, Y. Tsao,
  S.-D. Lin），页码 2305-2314 亦错，正确 PMLR 97:**2031-2041**
  （proceedings.mlr.press/v97/fu19b 核实）——已修。
- andrews2005interference：补 DOI **10.1109/MWC.2005.1421925**
  （v12 i2 p19-29 吻合；CrossRef 标题带 IEEE 栏目前缀，正文标题无误）。
- dankberg1998pcma：补 DOI **10.2514/6.1998-1398**（AIAA 1998 吻合）。
- ochiai2017multichannel（PMLR 70:2632-2641）核实无误；两条中文刊
  （JEIT 10.11999/JEIT251144、Modern Radar 10.16592/...）不在 CrossRef
  收录范围，维持原样；chen2019deep 会议 2019/论文集 2020 年份口径
  维持会议年（沿用 paper1 既有条目）。
- 教训：写作期凭记忆补的条目必须过 CrossRef——6 条新引里 4 条有错。

**cover letter**：`paper5/cover_letter.tex`（`bash build.sh cover` 编译，
2 页已目检）。按既定定位：measurement/protocol paper，三条贡献
（K 轴经验规律 / 协议与统计口径 / 受控探针归因+处方），主动声明
负面结果及其社区价值，衔接 PCMA/SIC 文献，可复现性声明，related
submissions 披露（paper1 @WPC、paper4 @ChinaComm，无共享结果）。
未套用 paper2/3 的"新方法"叙事模板（仅复用 LaTeX 脚手架与签名块）。

**AGENTS.md** 同步：paper5 状态行更新为当前口径，布局树补 paper5/ 条目
（含 tables/*.tex 由 make_tables.py 生成、勿手改的约定）。

---

## 2026-09-09 第四轮外部评审处置（投稿前终审）

评审含「期刊适配 + 6 条数据自洽 + 6 条论证缺口 + P2 合规」共 20 余条。
逐条独立核实（eval JSON 重算 + 代码回查 + 终版排版实测），裁决如下。

**驳回（评审不成立，有实证）**：
- **页数超限（36>20 desk-reject）**：驳回。`\documentclass[review,12pt]`
  是评审格式；实测 `final,3p,times` 终版编译 **14 页**，低于 AEÜ
  20 页上限。评审混淆了两种排版口径。
- **out of scope**：驳回。AEÜ 官方 scope 明确含 "signal and system
  theory, digital signal processing, communication theory and
  techniques, modulation"（多源核实），ML-for-PHY 算法稿在范围内；
  "desk-reject" 风险被夸大。但"零 AEÜ 自引"是弱 fit 信号（可选补强）。
- **摘要 200 词硬限**：无法证实（ScienceDirect 指南 403，Elsevier
  通行口径 ~250 词）。审慎起见摘要由 280 词修剪至 **259 词**。
- **AI 声明收窄为 "language editing"**：驳回。现措辞
  "drafting, editing, and polishing" 与事实相符，收窄反而不实。

**属实并已改稿（P0 六条全部属实，原始 JSON 复核确认）**：
1. −10 dB 处 slot SER 0.5905 vs baseline 0.5835（(b)，恶化 +0.70pp；
   specialist 0.5852，+0.18pp/0.0007std ≈2.6σ）——"buys nothing" 改
   "buys nothing, if anything a fraction-of-a-point penalty"（摘要/
   贡献点 1/§5.2/结论四处同步）；"bands sit on" 改 "hug"。
2. 摘要 K=2 "+0.35pp" 来自替换对照 (c)vs(a)——改报区间口径
   "+0.06 到 +0.40pp across contrasts"（四对照实测 0.345/0.400/0.110/
   0.056），公平加法对照 (d)vs(b)=+0.11pp 在摘要/贡献点 2/§5.3 均标注。
3. §5.3 K=1 "nothing moves (|Δ|≤0.01pp)" 与表矛盾（四列实为
   −0.01/−0.19/+0.26/+0.45）——改为"small and sign-unstable，至多
   4/5 同号，最大值主要由单 seed 驱动"。
4. "K=2 count acc ≈0.001→≈0.77" 是跨 seed 摘樱桃——改均值口径
   0.035→0.63（附逐 seed 极差 0.001–0.11 / 0.19–0.84），并补 tol-1
   accuracy 全程 0.97–1.00 的事实（occupancy 本来就能数准到 ±1，
   任务项做的是**阈值再校准**而非计数能力；阈值无关 AUC 列为
   future work）。
5. cell-level 置换检验（n=35 = 5 seed × 7 SNR，同 seed 内不独立）——
   §5.3 reading 1 重写：seed-level 检验明确为 cluster-correct 主检验，
   cell-level 降为 descriptive 并当场声明 anti-conservative；结论
   "statistically real" 改 "not statistically resolvable at five seeds"；
   Limitations 第五条同步强化。
6. recursive 剔除——保留 artefact 判定，Limitations 新增第七条：
   matched-pair baseline 重算才能完全闭环（需服务器 checkpoint 重评）。

**属实并已改稿（P1 文本层）**：
- §5.2 新增绝对口径：slot 输出 SI-SDR +0.1 dB / SIR −1.6 dB（干扰残留
  功率 ≈1.4× 目标功率）——比 SI-SDRi 更直接支撑 interference-limited。
- §5.5：v1 "exactly zero" 改 "almost exactly"（真载波在标称 2000 Hz
  ±10 Hz 内游走，data_generator_vark.py 确认 U(0,5)±5Hz）；v1 诊断降级
  为 "candidate cause"；v2 补 32-tap < 64-tap RRC 的部分修复保留意见。
- §5.5/探针：补 parity 句——oracle 同步下探针 val SER 0.150 仅**追平**
  K=1 波形路由基线 0.146，证明盲同步是失败的**必要**解释而非**充分**
  条件；K=2 探针留白。
- §6：K=3 负效应补两个未排除的平凡解释——losses.py:230 按 K 求均值
  导致 K=3 每源任务梯度仅 1/3（梯度稀释）；σ²=0.1 时 16QAM 最近邻
  d²/σ²≈4 已饱和（logit 饱和可独立解释 ser_soft 2.95→2.84 平台与
  平坦剂量响应；σ² 消融只到 0.2）。
- Limitations 新增第六条：SIR 全程固定 ≈0（w~U(0.4,0.6)），K 轴与
  总干扰功率/槽位稀疏度共变，定 K 扫 SIR 是下一步实验。
- cover letter 作者贡献与 CRediT 对齐（Jiang: experimental
  investigation → resources and validation）。
- highlights 第 1 条同步 "buys no bit-level gain at all"（≤85 字符/条）。

**待用户决策 / 需服务器**：
- 补实验（性价比序）：① 定 K=2 扫 SIR∈{0,5,10,20} dB（画相图，直接
  撑标题论断）；② K=2+oracle 同步探针（唯一可能的正面结果）；
  ③ recursive matched-pair 基线重算；④ occupancy 阈值无关 AUC；
  ⑤ per-K λ_ser=3 剂量响应 + σ²=1.0。
- ⚠️ 跨稿件风险：Data availability 指向公开 GitHub（实名+含
  paper4_open_world/），paper4 在 ChinaComm 双盲评审中——公开仓库
  可能已构成破盲渠道，需评估（暂存/转私有或接受风险）。
- 参考文献 26 条、零 AEÜ 自引；公式 3 个（终版 14 页有余量，可补
  SI-SDR/PIT/BER 形式化定义）——均未动，视期刊最终决定。

**验证**：main.pdf 37 页（review）/14 页（final 3p）编译干净（无 ??
无 overfull），全部新措辞在 PDF 文本层逐字核查落位；abstract.txt
259 词与 main.tex 同步；cover_letter.pdf 3 页。

---

## 2026-09-09 投稿决策执行（五项）

用户决策：① 补实验（SIR 扫描 / K=2 oracle 探针 / matched-pair 基线）；
② **GitHub 仓库暂停推送**（paper1–4 全部在审，paper4 双盲，避免破盲
渠道更新）；③ 坚持 AEÜ；④ 补形式化定义；⑤ cover letter 声明姊妹稿
可按需提供。

**已落地（本地）**：
- 形式化定义：新增 eq:sisdr（SI-SDR 投影定义）、eq:pit（变 K 内射分配
  枚举，4/12/24 per K，与 losses.py:74 `assignment_table` 核对）、
  eq:serber（SER + Gray BER/Hamming）。全文公式 3→6 个。
- AEÜ 文献：补 su2017underdetermined（Su/Shen/Wei/Deng, AEÜ vol.77
  pp.43-49, 2017, DOI 10.1016/j.aeue.2017.04.025，CrossRef 核实），
  引在 §2.1 经典欠定 BSS 处（时频稀疏路线）。CrossRef 全刊检索确认
  AEÜ 在"单通道通信信号分离"上没有更近的相关文献，只补这 1 条，
  不为 fit 堆砌边缘引用。
- cover letter：related submissions 段补 "the cited companion
  manuscript is available to the editor and reviewers on request"；
  作者贡献与 CRediT 对齐（Jiang: resources and validation）。
- 终版排版复核：final,3p,times 编译 16 页（+2 页来自新公式/文献），
  仍 ≤20。review 版 37 页，编译干净（无 ?? 无 overfull）。

**服务器补实验（后台 subagent 执行中）**：设计要点——
① SIR∈{0,5,10,20} dB 定 K=2 扫描：几何对称权重 w=0.5·10^(±SIR/40)，
   独立测试种子 88888（区别于主测试集 99999），评估现有 S2 四配置
   ×5 seed 的 checkpoint，不重训；
② K=2 oracle 同步探针：ProbeHead 扩展为双 slot 输出 + 2! PIT +
   星座掩码 CE；S variant 为**仅频率 oracle**（K=2 混合物 per-source
   相位纠缠，无法对齐——这本身是论点的一部分）；
③ recursive matched-pair 基线：在 recursive 自己的匹配子集上重算
   mixture baseline，直接检验"选择偏差伪影"判定。

---

## 2026-09-09 三项补充实验完成（服务器，当日完成）

**实验③ recursive matched-pair 基线 → 伪影实锤**：`eval_matched_baseline.py`
（复用 evaluate._match_and_measure，主测试集 seed 99999 同口径；recursive
SER 0.4477±0.0063 逐位复现 S1 作校验）。在 recursive 自己匹配上的源子集
上，未处理 mixture 的 SER = **0.4454±0.0062** ≈ recursive 的 0.4477
（recursive 反而差 0.0024）；按 SNR 逐 cell 两者同步。相对全源 baseline
0.5049 的"+5.7pp 增益"完全是漏检源不计分的选择效应。
→ 稿件 §5.1 从"排除为 artefact"升级为"matched-pair 控制实验直接证实"。

**实验① SIR 扫描（定 K=2，SIR∈{0,5,10,20} dB，测试种子 88888，
SNR 全网格 ×100，S2 四配置 ×5 seed 不重训）**：`eval_sir_sweep.py` /
`aggregate_sir_sweep.py` / `results/sir_sweep/`。以 (b) 为例：

| SIR | baseline | (b) SER | Δ(pp) | SI-SDRi | 强/弱源 SER |
|---|---|---|---|---|---|
| 0 | 0.5710 | 0.5629±.0016 | +0.82 | +3.2 | — |
| 5 | 0.5500 | 0.5478±.0006 | +0.22 | +3.6 | 0.426/0.669 |
| 10 | 0.4921 | 0.4923±.0002 | −0.02 | +3.9 | 0.273/0.712 |
| 20 | 0.4569 | 0.4658±.0033 | −0.89 | +3.0 | 0.193/0.738 |

pooled SER 随 SIR 下降但 per-cell baseline 同步下降，差距从未超过 1pp
且 SIR=20 反号；改善全部来自强源（干扰淡出），弱源恶化——不对称 SIR
只是把 cell 权重移向"本来就不需要分离"的源。**失配不是 SIR≈0 的特例，
贯穿整个扫描范围**（标题论断从 K 轴代理升级为 SIR 轴直接证据）。
caveat：SIR>3.5 dB 的 cell 对训练分布是 OOD（SI-SDRi 仍保持 +3~4 dB）。
→ 稿件 §5.1 新增段落 + tab_sir_sweep（make_tables.py 生成）。
注意 ser_mse 在 SIR=20 std=0.068（单 seed 异常），表只取 (b)。

**实验② K=2 oracle 同步探针 → 负面结果**：`probe_sync_head_k2.py`（双
slot 输出 + 2! PIT + 星座掩码 CE，30 epochs）。P（盲）= 0.7112 平台；
S（**仅频率 oracle**：双通道真载波下变频——K=2 混合物 per-source 相位
纠缠，相位对齐在原理上不可能）= **0.5888**，差于 mixture baseline
0.5611，更远差于波形路由 0.5475。K=1 的"oracle 同步救回头"不能外推
到 K=2：**约束不在频率搜索，而在相位纠缠下的联合检测本身**。
→ 稿件 §5.4(S3)、§6 处方、结论同步改写（"sync 必要但不充分"）。

**稿件同步**：§5.1 recursive 段与 SIR 扫描段、tab_sir_sweep、tab_s2_main
caption（cell-level 检验降为 descriptive）、tab_s2_perk caption（K=1
措辞）、tab_s3 caption（v1 降级为 candidate + K=2 探针）、§6 处方
（补 joint/SIC 检测结构）、Limitations 第六/七条（SIR OOD 范围、
K=2 探针仅频率 oracle + 30 epoch 平台）、结论、贡献点 1/5、
highlights（#1/#5）。main.pdf：review 40 页 / final 3p **16 页**（≤20），
编译干净，全部新措辞 PDF 文本层核查落位（注意 pdftotext 的 ﬃ 连字会
骗过朴素子串匹配）。

---

## 2026-09-09（终审轮）：投稿前全文自审 + 修正

自审方式：main.tex 全文通读 + 三路独立核查（子代理）——
(1) 五张表 164 个数字用逐 seed eval JSON 重算（population std、
2^5 全枚举 sign-flip、1e6 MC 翻转均精确复现），全部吻合；
(2) 公式/超参数/训练流程与代码逐行比对；(3) 27 条参考文献经
CrossRef/出版社官网逐条核实（无错误、无孤儿引用）。

**修正（全部已改稿并编译验证，PDF 文本层逐条落位核查）**：
- tab_s2_perk caption：p 值说明 "0.125 for 4/5 or 1/5" 错误——4/5
  格的 exact p 取决于各 seed 差值幅度（本表 7 个 4/5 格实测
  0.1875–0.5625，仅 1/5 恒为 0.125）。改为 "0.125 for 1/5,
  >=0.125 for 4/5"。（make_tables.py 改，表格重新生成。）
- §5.1 训练流程：ReduceLROnPlateau 的对象写错（论文写 validation
  loss，代码 train.py:403-447 实为 val SI-SDR；config (e) 为 val
  soft-SER）——已按代码改正。
- eq:sisdr：补零均值化说明（losses.py:53 / evaluate.py:121 均先
  去均值再投影，公式原来没写）。
- §5.4 per-modulation SER 四个数更新为当前 5-seed artifacts 值
  0.236/0.412/0.620/0.742（原 0.234/0.411/0.619/0.742 为 09-02
  快照）。
- §5.1/§6/tab_sir_sweep caption："flips sign at SIR=20 dB" 与表
  不符（10 dB 已为 −0.02）——改 "crosses zero near 10 dB, clearly
  negative at 20 dB"。
- §5.5 "carriers wander ±10 Hz" → "within −5/+10 Hz"（生成器实际
  为 U(0,5)+U(−5,5)，非对称）。
- §3.1/§5.5 RRC "64 taps" → "65 taps"（num_taps=64 参数实际生成
  t=−32..32 共 65 抽头；注意 paper1–4 的表述未同步改）。
- §4.3 "(a) λ_sep=1 only" 字面不成立（λ_occ=1、λ_cnt=0.1 全程
  常开）——§4.3 开头补说明，(a) 改 "nothing else"。
- count-head "~0.91 for all configurations" → "≈0.9
  (0.87–0.92)"（三处：§1 贡献点 4、§5.1、§5.3）。
- §6 "repaired from chance to 0.85" → "from 0.65 to 0.85 (from
  near-chance at K=2)"（0.65 并非 chance）。
- §5.1 计数指标补 "floored at 1"（evaluate.py:342 的 max(k_occ,1)）。
- §5.1 "SER spread ≤ 0.014" → "≈0.014"（实测 0.01404）。
- tab_s2_main caption 高 SNR 格区间 "10–20 dB, +0.2–0.5 pp per
  cell" → "15–20 dB, +0.2–0.3 pp in the cell means"。
- **新增披露（自审新发现，四轮外部评审均未抓到）**：训练 SNR 为
  U[−5,20] dB，测试网格含 −10 dB ——该 cell 在训练分布外，而摘要/
  引言/结论的卖点 "+8.4 dB at −10 dB buys nothing" 正落在其上。
  Limitations 新增 Seventh（原 Seventh 顺延为 Eighth）声明该 cell
  为外推工作点、仅作最大增益格使用、pooled 结论不依赖它；§5.1
  同处加交叉引用。
- 摘要：SI-SDR/SER/BER/pp 首次出现展开；"buys nothing, if anything
  a fraction-of-a-point penalty" 语法修正为 "buys nothing—if
  anything it costs a fraction of a point"；hallucination
  "0.15→0.05" → "0.15→0.05–0.06"（(c)=0.060，仅 (d)=0.049）。
  摘要 252 词。
- 参考文献作者截断统一：ma2023novel / luo2023single / edsnet2026
  的 et al. 展开为全作者列表（CrossRef/期刊官网查实：Ma Hao 等 5
  人；Luo Weilin 等 6 人；Zhang Jing 等 4 人含 Jin Benzhou）。
- cover_letter.tex："buys nothing" 补 penalty nuance（与摘要口径
  一致）；"Two companion manuscripts" → "Two related companion
  manuscripts"（paper2/3 也在审，避免被读成穷举）。
- config.py：test_n_per_cell 默认 200→100（论文与全部 run 脚本
  实际均用 100；消除复现者不带 --n_per_cell 时的口径偏差）。

**编译验证**：main.pdf 40 页（review 格式），无 ?? 引用、无未定义
控制序列，仅 1 处 1.19pt overfull（可忽略）；cover_letter.pdf 3 页。
所有新措辞经 PDF 文本层逐条核查（pdftotext 连字/断行需归一化）。

**遗留（不阻塞投稿）**：GitHub 仓库按决策暂不推送（paper1–4 在审，
paper4 双盲）；投稿时 main.tex/cover letter 的代码可得性声明需
配套一次推送决策。nong2026openworld 被引 6 次仍在审（cover letter
已声明可按需提供）。
