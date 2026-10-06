我按 **IEEE Transactions on Signal Processing（TSP）这一级别期刊**的标准通读了这篇 13 页稿件，并重点检查了理论推导、算法逻辑、实验设计、统计有效性以及结论是否超出证据支持范围。总体判断比较明确：

## 总体审稿结论：Major Revision

这篇论文**有一个值得发表的问题意识和一个相当清楚的主线**：不要把 SC-BSS 的目标仅仅停留在 waveform fidelity，而应该观察“分离后能否真正完成通信检测”；作者进一步发现，在共频、单通道、近等功率条件下，独立分离—同步—检测会受到结构性限制，因此把载波同步和联合检测重新放回混合信号域。这个思路是成立的，而且 E3→E4 的实验链条具有一定说服力。

但是，以 TSP 标准衡量，目前版本还有几处**不是文字润色，而是会直接影响论文科学结论成立与否的硬伤**。其中最严重的是：

> **CRB 部分的频偏分布与前面的信号生成模型不一致；Phase identifiability 的命题表述过强；Theorem 2 和 Proposition 3 的理论支撑不完整；实验中又存在 genie、single-seed、matched-pairs-only 等评价设计问题。**

因此我目前不会建议接收，也不建议 Minor Revision。比较合理的是 **Major Revision**。如果这些核心问题无法补齐，我认为 TSP 编辑有理由直接拒稿。

---

# 一、这篇论文真正有价值的地方

### 1. “waveform quality ≠ communication quality”的问题切得比较准

论文没有简单地再造一个 separator，而是问了一个对通信接收机更重要的问题：**SI-SDR 很好，为什么 SER 仍然很差？**

论文的 E3 实际上把这一问题展示得比较直接：K=1 时 oracle receiver 下 pooled SER 为 0.1454，而 K=2、3 时分别变成 0.5534 和 0.5858，即使这里使用的还是 oracle carrier/phase receiver。这个实验确实能够证明“分离波形质量并不自动转化为检测性能”。

这个 framing 比单纯“提出一个更复杂的网络”有研究价值。

### 2. E3→E4 的因果链条是目前论文最好的部分

论文不是直接报一个 V4 比 baseline 好，而是设计了：

`separate → blind sync → joint detector`

与

`raw mixture → joint sync/detection`

以及

`memoryless → ISI-aware → oracle-carrier`

的一系列对照。

尤其 V1/V3/V4 形成了一个比较漂亮的 error budget：

* V1：0.5380
* V4：0.5299
* V3：0.4104

这让作者能够把剩余误差分成“模型失配”和“频率获取”两个部分。 

从审稿人的角度看，这比单纯做一个 ablation table 要有意义。

### 3. 作者对失败案例的披露比较诚实

例如文章明确承认：

* 16QAM 在很多场景下几乎没有 gain；
* K=3 acquisition 会锁到 phantom pair；
* blind modulation classification 很差；
* timing recovery 没有处理；
* V4 需要约 2–2.3 s/burst CPU 时间；
* M-fold phase ambiguity 仍然靠 genie resolution。

这种写法比“所有模块都有效”的论文可信度高。

所以我认为这篇论文**不是没有结果，而是目前理论和评价口径没有完全收紧**。

---

# 二、最重要的问题 1：Proposition 3 的频偏分布与 System Model 明显不一致

这是我认为全文目前最需要优先修正的问题。

论文在 System Model 中定义：

$$
u_k\sim U(0,5),\qquad \epsilon_k\sim U(-5,5),
$$

因此每个源的 residual offset 是

$$
\delta_k=u_k+\epsilon_k,
$$

其支持区间是：

$$
[-5,10]\text{ Hz}.
$$

论文随后在 Proposition 3 中却写：

> “Under the benchmark’s offset law \(|\Delta f|\) triangular on [0,10] Hz ... median 1.82× and 90th percentile 10.5×.”

但根据前面的生成模型，两个源的 offset 差：

$$
\Delta f=\delta_1-\delta_2
$$

至少可以达到 \([-15,15]\) Hz，因此

$$
|\Delta f|\in[0,15]\text{ Hz},
$$

而不是 \([0,10]\) Hz。前面的 benchmark 定义可以直接在论文中找到。

这不是一个小笔误，因为 **1.82× median 和 10.5× percentile 恰恰是 Proposition 3 的核心 quantitative result**。一旦 offset law 不一致，这两个统计量就失去了明确含义。

### 我建议必须这样修

正文明确写出：

$$
\delta=u+\epsilon
$$

的卷积密度，然后严格推导

$$
|\delta_1-\delta_2|
$$

的 PDF/CDF。

如果实验实际上使用的是另一个 offset distribution，那么应该修改 System Model，而不是修改理论去迁就结果。

同时建议重新报告：

* mean；
* median；
* 90th percentile；
* 95th percentile；
* \(\Delta f<1/T_{\rm burst}\) 的概率。

否则 Proposition 3 目前不能作为一个严谨的“benchmark-specific theoretical result”。

---

# 三、最重要的问题 2：Proposition 2 的“identifiability”表述过强

论文写的是：

> “Discrete alphabets restore joint identifiability.”

并进一步说 carrier phases 因此 identifiable up to \(Z_{M_1}\times Z_{M_2}\)。

这个表述对 TSP 来说太强。

事实上，论文自己已经给出了一个直接反例：

> QPSK 在 \(\phi=0^\circ,90^\circ,180^\circ,270^\circ\) 时 \(d_{\min}(\phi)=0\)。

例如 QPSK+QPSK、等幅、\(\phi=0\) 时，不同的 symbol pair 可以产生完全相同的混合点。这意味着**离散 alphabet 并不能无条件恢复 source pair 的唯一可辨识性**。

此外至少还存在：

1. source permutation ambiguity；
2. alphabet rotational ambiguity；
3. zero-contact / constellation collision；
4. equal-alphabet + equal-power 情况下的额外对称退化。

因此更严谨的说法应该接近：

> Under generic non-colliding channel coefficients and relative phases, discrete finite alphabets reduce the continuous gauge freedom to a finite symmetry class, but do not guarantee global source identifiability at collision phases or under source-permutation symmetries.

也就是说，建议把 **“restore identifiability” 改成 “reduce the continuous ambiguity / yield generic identifiability under non-degenerate conditions”**。

这会让理论明显更稳。

---

# 四、最重要的问题 3：Theorem 2 是核心定理，但没有真正给出证明

这是全文理论完整性上的一个明显缺口。

正文把 Theorem 2 称为：

> “Exact separate-detection floor.”

并给出 0.488 的精确 floor。

但 Appendix A 的标题却只有：

> “Proofs of Theorem 1 and Propositions 1–2”

也就是说：

* Theorem 1：有证明；
* Proposition 1：有证明；
* Proposition 2：有 sketch；
* **Theorem 2：没有证明；**
* **Proposition 3：正文给出了结果，但 Appendix 中没有相应的完整推导。**

现阶段这和“three theoretical findings organise the design”以及“exact CRB inflation analysis”这种论文定位不完全匹配。

### 更关键的是，我建议作者重新核对 0.488 这个“exact floor”

论文使用的是一个有限 phase grid，并且正文说：

> “12 fixed phases offset from the zero contacts”



我按式 (6) 的高 SNR 极限重新检查时，**对于连续均匀 \(\phi\sim U[0,2\pi)\) 的 phase ensemble，结果并不自然等于 0.488；更接近 0.5。**

所以我非常怀疑：

> **0.488 很可能是所选 12 个 phase points 下的数值，而不是严格意义上的连续 phase ensemble exact floor。**

这必须澄清。

建议把以下三个量分开：

$$
P_{\rm sep}(SNR\to\infty\mid\phi)
$$

$$
E_{\phi}[P_{\rm sep}(\phi)]
$$

以及

$$
\frac1{12}\sum_{i=1}^{12}P_{\rm sep}(\phi_i).
$$

如果 0.488 对应第三个，就绝对不能在 abstract 中写成不加限定的

> “exact separate-detection floor = 0.49”。

这属于**定理适用域和实验数值混淆**。

---

# 五、最重要的问题 4：Joint MAP 并不是严格意义上的“最优 per-source detector”

论文式 (3) 做的是：

$$
(\hat c_1,\hat c_2)
=
\arg\min_{c_1,c_2}
|r-c_1-a_2c_2|^2.
$$

也就是 **joint ML/MAP pair detection**。

但论文最后评价的是 **per-source SER**。

这两者并不完全等价。

真正针对 source-1 symbol error 的 Bayesian marginal MAP 应该是：

$$
\hat c_1
=
\arg\max_{c_1}
\sum_{c_2}
p(r|c_1,c_2)p(c_2).
$$

论文自己已经观察到：

> at SNR \(\lesssim5\) dB，joint ML per-symbol decisions 甚至略差于 separate detection，因为 joint rule optimise the pair, not the marginals.



这恰恰说明这里存在一个很重要的问题：

**你是在比较“pairwise MAP”与“per-stream SER”，还是在比较真正的最优 marginal detector？**

### 必须增加一个 baseline

至少增加：

* separate MAP；
* joint pair ML；
* marginal MAP；
* 如果考虑 ISI，则增加 sequence MAP / BCJR / Viterbi 类 baseline。

否则“joint detection 是 fundamental remedy”这个结论还不够严谨。

---

# 六、最重要的问题 5：你把 Hard-EM / Classification-EM 称为 EM，需要更加准确

IV-C 中写：

> E-step = joint decision
> M-step = LS refit



严格说，这不是通常意义上的 soft EM：

$$
q_{ij}
=
p(c_1=i,c_2=j|z,A)
$$

然后根据 posterior responsibility 做 M-step。

你的做法更接近：

* hard EM；
* classification EM；
* alternating ML / coordinate ascent。

而 V4 又进一步加入：

* frequency grid search；
* converged EM；
* NLS polish；
* multiple restart。

所以它实际上是一个**离散变量 + 连续参数的联合非凸优化过程**。

论文虽然说：

> “it is energy-monotone”

但目前没有充分说明：

1. 每次 update 是否保证目标函数不下降；
2. frequency grid move 后是否严格重新优化 latent variables；
3. restart 后如何定义最终 objective；
4. NLS polish 是否保持单调；
5. “convergence”具体是什么 criterion。

论文后面确实说明 truncated EM 会出现 phantom offset，而且每一个 frequency move 都重新 convergence。

这是很好的经验结果，但建议把算法正式命名为：

> **ECM / classification-EM based joint synchronization and detection**

并给出一个完整 Algorithm 1。

这会比当前的“EM”表述更加经得住理论审稿。

---

# 七、最重要的问题 6：CRB 与实际 BlindCarrierSync 的比较方式不够严格

论文把单音 CRB 与实际 BlindCarrierSync error 对比，并且说某些情况下接近 CRB。

但 Fig. 3 的说明里明确写了：

> “Measured ... error ... excluding outliers.”



这对于 CRB 比较是一个问题。

CRB 针对的是估计量的 bias/variance 条件，而你的算法在低 SNR 或弱谱线情况下有明显 failure/outlier。

如果删掉 outliers：

$$
{\rm RMSE}_{\rm inlier}
$$

可能非常漂亮，但这不等价于完整 estimator performance。

### 建议同时给出

$$
{\rm Bias}(\hat f)
$$

$$
{\rm Var}(\hat f)
$$

$$
{\rm RMSE}(\hat f)
$$

以及

$$
P(\text{acquisition failure})
$$

和

$$
P(|\hat f-f|>\tau).
$$

特别是 16QAM 和 8PSK 的 failure probability 应该明确报告。

否则“within 1.1–1.8× CRB”容易被审稿人认为是经过筛选后的 favorable statistic。

---

# 八、最重要的问题 7：E3 的 “matched pairs only” 会产生评价偏差

E3 中的流程是：

> occupied slot → threshold > 0.5 → Hungarian matching → matched pairs only → SER



这对普通 separation evaluation 可以接受，但对本文的核心命题——**完整通信接收链路**——就有问题。

例如：

* 一个 source 没分出来；
* occupancy 低于 0.5；
* 多了一个 phantom slot；

最后“matched pairs only”很可能直接把这些情况排除在 SER 统计之外。

那么你的 SER 实际上是：

> **conditional SER given successful source-slot matching**

而不是完整 receiver SER。

这与论文反复强调“bits are the deliverable”并不完全一致。

### 建议必须补

至少报告：

* source-count accuracy；
* miss probability；
* false alarm slot probability；
* slot assignment accuracy；
* conditional SER；
* **end-to-end SER including missed-source penalty**；
* BER；
* 最好再给 GMI 或 mutual-information-related metric。

否则当前 0.5534/0.6078 这类数字存在一定“selection conditioning”。

---

# 九、最重要的问题 8：Genie 太多，导致“blind”这个词需要收紧

目前 pipeline 虽然叫 blind，但实际上有不少已知信息：

### 已知

* \(K\in\{1,2\}\)；
* modulation 已知；
* nominal carrier \(f_0\) 已知；
* CFO range 已知；
* timing 已知；
* symbol grid 已知；
* RRC 参数已知；
* E1 还有 genie M-fold phase resolution；
* oracle-carrier receiver 用于上界。

论文自己也明确把“known modulation”作为整个 pipeline 的 operating assumption。

因此，我建议避免把全文概括成完全意义上的：

> “blind joint receiver”

而更严谨地使用：

> **training-free / model-based / modulation-conditioned / semi-blind**

具体采用哪个名称由作者决定，但需要让 reader 一眼知道：

**这是一个强先验条件下的 blind-ish receiver，而不是完全未知环境中的 blind communication receiver。**

---

# 十、最重要的问题 9：Proposition 3 的 CRB 模型和真实通信波形之间存在明显 model gap

CRB 推导使用的是 two-tone：

$$
Ae^{j(2\pi fn/f_s+\phi)}
$$

然后考虑两个 unknown-frequency tones。

但实际 benchmark 是：

* PSK/QAM；
* RRC；
* multipath；
* 3-tap fading；
* symbol transitions；
* unknown complex gains。

也就是说，理论中的 FIM 并不是实际 communication waveform 的完整 FIM。

而正文却把 Proposition 3 用来指导整个 architecture：

> “so per-source synchronisation belongs after separation...”



这个 architectural conclusion 可能仍然是正确的，但理论应该明确称为：

> **pure-tone surrogate / information-theoretic proxy under the simplified observation model**

而不要让读者误以为这是实际 SC-BSS communication signal 的 exact CRB。

更重要的是，FIM 到底把哪些 nuisance parameters 当成 unknown？

目前至少需要明确：

$$
\{f_1,f_2,\phi_1,\phi_2,A_1,A_2\}
$$

中哪些是 estimated、哪些 fixed。

如果 amplitudes/gains 是实际未知参数，那么 CRB 应该把它们放进 nuisance parameter block，再做 Schur complement。

---

# 十一、实验最大的现实问题：benchmark 太“定制化”

这是 TSP 审稿人非常容易抓住的一点。

当前 benchmark 是：

* 256 symbols；
* 16 samples/symbol；
* 0.256 s；
* \(f_0=2\) kHz；
* CFO \([-5,10]\) Hz；
* weights \(U(0.4,0.6)\)；
* SIR 接近 0 dB；
* 3-tap channel；
* timing 完全固定；
* 无 fractional timing；
* 无真实采集数据；
* K=2 是核心；
* modulation set 固定；
* RRC roll-off 固定 0.35。



所以当前最强的结论实际上应该是：

> **在本文构造的近等功率、短 burst、同步采样、3-tap fading、已知 modulation 的 K=2 benchmark 上……**

而不能自然扩张成：

> “SC-BSS generally should detect on raw mixture and separation should only be used for counting/localisation.”

后者明显超出了当前证据。

论文自己也已经承认：

> true source streams + same joint loop can reach SER 0.041.

这其实说明：

**不是“separation inherently bad”，而是“current separator's distortion breaks the assumed linear mixture model”。**



这是两个完全不同的命题。

---

# 十二、因此，论文最重要的 conclusion 建议改弱一点

现在的表述：

> “Separation ... serves counting and localisation; detection belongs on the mixture”

过于普适。

更严谨的结论应该是：

> **For the studied K=2 benchmark and the tested separator quality, joint synchronization/detection on the raw mixture outperforms the slot-aided separated-input route. Separation may remain beneficial when its output preserves the underlying linear mixture model sufficiently well.**

这个版本反而更科学。

因为你自己的 true-source control 已经证明了这个边界。

---

# 十三、K=3 的结论也需要收紧

当前论文用了比较强的表述，例如：

> “K=3 fails at acquisition.”

实际上你的 experiment 是：

> 某一种 sequential extraction initialization + K=2 ECM + residual M-th-power acquisition + 3-source phase-grid EM

在 K=3 上失败。

这可以证明：

> **the proposed K=3 acquisition strategy fails**

但不能证明：

> **K=3 joint detection is fundamentally infeasible.**

后者需要更广泛的 acquisition strategies 或 impossibility analysis 才能成立。

建议全文统一改成：

> **the tested K=3 acquisition strategy fails / is unreliable**

而不是把它写成 K=3 本身失败。

---

# 十四、实验统计性还不够 TSP

E1 每个 modulation/SNR cell 大约只有：

> ≈25 bursts



而 E3/E4 大量结果使用固定：

> `test seed = 99999`

同时 V0/V2 是 5 个 checkpoint，V1/V3/V4 是 deterministic。

这对工程实验可以接受，但 TSP 审稿需要更强的 statistical evidence。

尤其论文强调：

> 1.1 SER points

这种很小的 difference 时，应该给出：

* 95% CI；
* paired bootstrap；
* 多个 test seeds；
* per-burst confidence interval；
* 最好对 paired comparisons 做统计检验。

否则“0.530 vs 0.553”到底是稳定 improvement 还是 benchmark-specific realization effect，读者无法完全判断。

---

# 十五、0.530 vs 0.553 的“改进幅度”其实没有论文文字里那么大

V4 相对 waveform route：

$$
0.5534-0.5299=0.0235
$$

只有约 **2.35 percentage points**。

这当然可能是有意义的，但是论文应该避免给人一种“dramatic improvement”的印象。

特别是：

* 16QAM-involving pair 基本没有 gain；
* 主要 gain 集中于 PSK；
* V4 相对于 V1 只有约 0.8 point。

论文其实自己已经给出了这些信息。

所以我建议把贡献表述成：

> **structurally meaningful but quantitatively modest improvement**

而不是把重点放在“beats the oracle-received waveform route”这种容易被 reviewer challenge 的句式上。

---

# 十六、baseline 还不够强

这是我认为第二个很现实的问题。

你当前真正的 winner V4 是一个**model-based joint multiuser detector**。

但论文对比体系主要是：

* 自己的 SlotSepNet；
* 自己的 blind synchronizer；
* 自己的 ECM variants。

这会导致 reviewer 问：

> “你证明的是你的方法好，还是证明 classical model-based joint detection 本来就应该这样做？”

特别是论文自己引用了 classical multiuser detection [16]，以及 paired-carrier / interference cancellation [17], [18]。

因此至少建议加入：

### Receiver baseline

* separate ML；
* marginal MAP；
* joint ML with true carriers；
* joint ML with estimated carriers；
* classical SIC；
* MMSE/SIC；
* EM without ISI；
* EM + ISI。

### BSS baseline

论文已经引用了多种 SC-BSS 方法，但没有给出足够完整的 quantitative comparison。

而且 2024–2026 年这一方向仍有新的 complex-domain、长序列以及联合收发/混合模型路线，至少需要在 related work 中重新定位你的 contribution，而不是只强调早期 SC-BSS。比如 2024 的 complex-domain SC-BSS、2026 的长序列/混合知识数据方法和 joint transceiver 方向都值得明确区分。([IEEE Xplore][1])

---

# 十七、建议增加的实验，我认为不需要很多，但这 6 类很关键

如果我是 handling editor，我会把 revision requirements 大致收敛到下面几项。

### Experiment A：修正频偏 distribution

重新推导 \(|\Delta f|\) 的真实分布，并重新计算 Proposition 3 的：

* median；
* 90th percentile；
* CRB curve。

这是必须的。

### Experiment B：补充 detector optimality baseline

至少增加：

$$
\text{Separate MAP}
$$

$$
\text{Joint ML}
$$

$$
\text{Marginal MAP}
$$

最好加入 sequence-aware baseline。

这会直接解决本文理论叙述中的一个漏洞。

### Experiment C：重新做完整统计评价

至少：

* 5–10 个独立 test seeds；
* 95% bootstrap CI；
* paired burst comparison；
* SER + BER；
* 不删除 outlier 的 frequency RMSE。

### Experiment D：不要只报告 matched-pair SER

把：

* missed source；
* false source；
* count accuracy；
* assignment failure

全部纳入 end-to-end metric。

这对于一篇以“receiver deliverable is bits”为核心命题的论文尤其重要。

### Experiment E：至少做一组真正的 robustness sweep

最少建议：

$$
SIR\in\{-10,-5,0,5,10,15\}\ {\rm dB}
$$

以及：

* CFO range；
* burst length；
* channel length；
* timing offset/fractional delay。

这四项里至少补两项。

### Experiment F：把 conclusion 改成 benchmark-conditioned

特别是把：

> separation should serve counting/localisation, not detection

改为：

> **under the current separator quality and tested benchmark**

这会显著降低 reviewer 对过度结论的攻击空间。

---

# 十八、还有几个次一级但值得修改的问题

### 1. Theorem 1 的 union bound 应进一步讨论高 SNR / zero-distance 情况

论文已经指出 phase mean 在高 SNR 时被 zero-contact neighborhoods 主导。

建议更清楚区分：

* pointwise bound；
* phase-averaged bound；
* median-phase；
* mean-phase。

否则“bound tracks simulation”在不同统计口径下容易混淆。

### 2. 16QAM 的 M-th power 说明值得补充更系统的解释

文中指出 fourth-power line occasionally cancels，这是实际很重要的限制。

建议增加一个 short theoretical explanation，而不仅是 empirical observation。

### 3. Complexity 需要更完整

现在给了：

> 2–2.3 s/burst CPU；full grid 24 min。



建议进一步给：

* CPU model；
* average EM iterations；
* number of frequency hypotheses；
* complexity versus \(N\)；
* complexity versus \(K\)。

这样 reader 才知道这个方法能否扩展。

### 4. “exact” 这个词应该克制

全文多处使用：

* exact floor；
* exact CRB inflation；
* exact identifiability。

其中一些确实是 exact，但一些其实是：

* simplified-model exact；
* finite-grid evaluation；
* generic identifiability；
* numerical FIM inversion。

建议把 “exact” 按适用条件严格限定。

---

# 十九、从“顶级期刊审稿人”的角度，我会怎样评价这篇论文

| 维度            | 我的判断                            |
| ------------- | ------------------------------- |
| 问题重要性         | 较高                              |
| 核心思想          | 有价值                             |
| 理论新颖性         | 中等，有一定亮点                        |
| 方法创新性         | 中等，主要是模型驱动 receiver integration |
| 实验逻辑          | 较强                              |
| 实验规模          | 偏小                              |
| benchmark 真实性 | 偏弱，强 synthetic assumption       |
| 理论完整性         | 目前不足                            |
| 统计严谨性         | 不足                              |
| baseline 完整性  | 不足                              |
| 结论边界          | 偏强                              |
| 可复现性          | 有积极信号，但我本次未独立验证代码仓库             |
| TSP 当前可接受程度   | **Major Revision**              |

---

# 二十、如果我是审稿人，我会给编辑的最终意见

可以直接写成下面这样：

> **Recommendation: Major Revision**
>
> This manuscript addresses an important and underexplored issue in single-channel blind separation of co-frequency communication signals: waveform separation quality does not necessarily translate into downstream detection performance. The manuscript has a coherent receiver-oriented narrative, and the E3/E4 experiments provide useful evidence that independently synchronizing separated slots can be structurally inferior to performing joint synchronization and detection on the raw mixture. The ISI-aware ECM formulation and the decomposition of the remaining error budget are also potentially valuable.
>
> However, several issues currently prevent the manuscript from meeting the rigor expected by IEEE TSP. Most importantly, the frequency-offset distribution used in Proposition 3 appears inconsistent with the offset-generation law specified in the system model, directly affecting the reported CRB inflation statistics. In addition, the identifiability proposition is stated too strongly because discrete alphabets do not guarantee global identifiability at constellation-collision phases or under source symmetries. Theorem 2 lacks a complete proof, and the reported “exact” high-SNR separate-detection floor needs to be carefully distinguished between finite phase-grid and continuous phase-ensemble averages. The receiver comparison also evaluates joint pairwise ML using per-stream SER, without a marginal-MAP baseline. Finally, the end-to-end experiments rely heavily on genie-assisted phase resolution, matched-pair-only scoring, a fixed test seed, a highly controlled synthetic benchmark, and known modulation assumptions.
>
> I therefore recommend major revision. The paper could become a strong contribution if the theoretical claims are tightened, the CRB/frequency model is corrected, the receiver baselines and statistical evaluation are strengthened, and the conclusions are explicitly restricted to the demonstrated operating regime.

## 最后一句比较直接的判断

**这篇论文不是“没东西”的稿件，恰恰相反，它的核心故事是比较完整的；现在真正的问题是“有价值的实验故事”前面挂着几处不够严谨的理论表述和评价设计。**

其中我认为最危险的三个点，按优先级排序是：

**① Proposition 3 的 offset distribution 不一致；
② Proposition 2 的 identifiability 结论过强；
③ Theorem 2/Proposition 3 的理论证明与 phase-ensemble 定义不够完整。**

这三个问题如果修好，再补 **marginal-MAP + 多 seed/CI + 完整 end-to-end scoring**，论文的科学可信度会提升一个层级。反过来，如果保持现状，我认为 TSP 审稿中被以“理论结论条件不严谨 + 实验 benchmark 过度定制”为理由拒稿的风险很高。

[1]: https://ieeexplore.ieee.org/document/10490090/?utm_source=chatgpt.com "Single-Channel Blind Source Separation in Wireless Communications: A Complex-Domain Deep Learning Approach | IEEE Journals & Magazine | IEEE Xplore"
