我按 **IEEE Signal Processing Letters（SPL）审稿人**的标准通读了这篇稿件，包括正文、图和表。总体判断是：

> **论文的问题意识很好，核心现象也很有价值，但以目前版本投稿 SPL，我会给出 Major Revision（大修）。**
>
> 最大的问题不是“结果不好”，而是**目前还没有把“一个具体代码 bug 的案例”充分提升为一个具有信号处理方法学意义、可推广、可复现的研究结论**。

稿件的核心贡献是发现 `np.repeat` 与 `np.tile` 导致 per-source SNR 标签与 score dump 错位，并证明这种系统性错误可以制造出看起来非常 convincing 的 OOD/SNR 结论；修正后六种 scorer 基本回到随机水平。这个核心故事是清楚的。

---

# 一、总体审稿结论

### Recommendation：Major Revision

我会给出类似下面的总体评价：

> This manuscript identifies an important and under-appreciated evaluation failure mode in SNR-conditioned open-set detection for single-channel blind source separation. The repeat-vs-tile mismatch is convincingly demonstrated to generate systematic and highly plausible performance artifacts. The deployment-oriented evaluation protocol is also potentially useful. However, the current manuscript does not yet sufficiently establish the generality of the proposed evaluation pitfall, and several important experimental and methodological details are missing. In particular, the definitions of per-source SNR, data ordering, score calibration, statistical uncertainty, source-separation quality, and scorer-specific hyperparameter selection need to be made substantially more rigorous. Moreover, the causal claim that the separation objective itself fails to create a known/unknown margin is currently stronger than the evidence supports.

**核心意思就是：论文值得做，但需要从“发现了一个 bug”升级成“建立了一套针对该类评测错误的系统方法学分析”。**

---

# 二、我认为这篇论文最有价值的地方

## 1. 论文的故事线非常完整

这篇文章实际上形成了一个很漂亮的闭环：

**错误标签**
→ **SNR 与 score 错位**
→ **人工制造 SNR separability**
→ **产生貌似合理的 OOD complementarity**
→ **多 seed / robustness / SNR estimation 都无法发现错误**
→ **重新对齐标签**
→ **全部性能回到 chance**
→ **提出 deployment-faithful evaluation protocol**

这一点是论文最强的地方。

尤其是你给出了非常具体的错误机制：

> score 按 source slot 进行 stacked/tile-wise 存储，而 SNR label 使用 interleaved `np.repeat`。

并且进一步给出了 `114/96/100/74` 这种违反预期 protocol 的 per-bin count 作为审计证据，这比单纯说“发现了代码错误”强很多。

---

## 2. “为什么传统 robustness check 也发现不了”这个点很好

这是我认为最有潜力成为论文真正贡献的部分。

稿件指出：

> 多 seed 只是把**系统性错误重复平均**；

> signal perturbation 改变的是信号，而不是 label geometry；

> SNR estimation noise 只是模糊错误产生的 SNR signal，所以反而会形成“graceful degradation”。

这个解释非常有意思，而且具备方法学意义。

如果这个部分继续做深，我甚至认为它比单纯的 `repeat vs tile` 本身更有发表价值。

因为 reviewer 真正关心的是：

> **“为什么一个错误能够通过通常的数据科学验证流程？”**

而不仅仅是：

> “某一行代码写错了。”

---

## 3. 修正后的结果并没有“美化”，这是优点

很多论文发现实验 bug 后，会把论文重新包装成“修正以后方法还是很好”。

你们这里恰恰相反：

> corrected routed AUROC = **0.498 ± 0.020**

> oracle = **0.549 ± 0.031**

而且 calibrated threshold 下 unknown rejection 只有：

> **0.058 ± 0.038**

这种结果从科研诚信和方法学角度看反而很有价值。

---

# 三、最大的审稿问题：现在更像“bug report”，还没有完全成为“signal processing methodology paper”

这是我最重要的一条意见。

现在核心事实是：

> 你发现了一个具体的 `repeat/tile` 数据排列错误。

但 SPL reviewer 很容易问：

### “为什么这个错误值得发表？”

因为从严格意义讲：

> `np.repeat` 写错是 implementation bug，不是新的 signal processing theory。

所以必须进一步证明：

### 这个问题代表的是一类普遍存在的 evaluation failure mode。

目前稿件虽然在 conclusion 中给出了 checklist，但论证仍然主要停留在这一个实验 pipeline。

### 建议增加一个非常重要的小节：

**Generalization of the Label-Alignment Failure**

至少把这种错误抽象成：

$$
\text{Data index}
\rightarrow
(\text{mixture},\text{source},\text{SNR},\text{class})
$$

而 score tensor 和 label tensor 分别采用不同 flatten/order：

$$
\pi_{\text{score}}\neq\pi_{\text{label}}
$$

最终：

$$
y_j = y_{\pi_{\text{label}}(j)}
$$

而真实 score 应该对应：

$$
s_j=s_{\pi_{\text{score}}(j)}
$$

当：

$$
\pi_{\text{score}}\neq\pi_{\text{label}}
$$

并且某个条件变量 \(c\) 本身影响 score distribution 时，

$$
p(s|c_1)\neq p(s|c_2)
$$

就可能人为产生：

$$
p(s|y=0,c)\neq p(s|y=1,c)
$$

从而让 AUROC 看起来显著高于随机。

这就把论文从：

> **“我们有一个 repeat/tile bug”**

提升成：

> **“我们发现了 conditioned OOD evaluation 中一种由 indexing permutation 引入的系统性 confounding mechanism。”**

这两者在审稿人眼中的层次完全不同。

---

# 四、第二个重大问题：per-source SNR 的定义不够清楚

这是信号处理 reviewer 很可能第一眼就问的问题。

你们写的是：

$$
y(t)=\alpha_1s_1(t)+\alpha_2s_2(t)+n(t)
$$

然后使用 SNR grid：

$$
\{-10,-5,\ldots,20\}\mathrm{dB}
$$

但是目前没有非常严格地定义：

### 这个 SNR 到底是什么？

是：

$$
\mathrm{SNR}_i=
10\log_{10}\frac{E|\alpha_i s_i|^2}{E|n|^2}
$$

还是 mixture SNR：

$$
\mathrm{SNR}_{mix}=
10\log_{10}\frac{E|\alpha_1s_1+\alpha_2s_2|^2}{E|n|^2}
$$

还是经过 channel 后定义？

这是本文标题中 **Per-Source SNR** 最重要的概念之一。

尤其你们后面直接使用：

> `np.repeat(SNRmix, 2)`

这一点更加需要解释。

### 我建议论文直接增加一个正式定义：

> For source \(i\), the operational SNR is defined as ...

然后明确：

* source power 如何归一化；
* \(\alpha_1,\alpha_2\) 是否相同；
* multipath channel 是否包含在 SNR 定义中；
* SIR 与 SNR 如何区分；
* kk/ku/uu 中的 SNR 如何生成；
* 一个 mixture 的两个 source 是否保证相同 SNR。

这是**必须补**的。

---

# 五、第三个重大问题：数据生成描述严重不足以支撑复现

目前 Section II 虽然给出了：

> RRC、3-tap multipath、AWGN、T=4096、carrier 2000/2005 Hz、SNR grid、modulation sets、69K parameter CNN 等信息。

但是对一个强调“evaluation methodology”和“reproducibility”的论文来说还不够。

至少还缺：

### 信号参数

* sampling rate \(F_s\)
* symbol rate
* RRC roll-off factor
* samples/symbol
* channel coefficient generation
* channel normalization
* phase offset
* timing offset distribution
* CFO distribution
* SIR distribution
* source power ratio

### 数据规模

例如：

* 每个 SNR 有多少 mixtures？
* 每个 modulation 有多少 source？
* kk/ku/uu 分别多少？
* known / unknown 总样本数？
* 每个 source 最终产生多少 score？

稿件只明确写出：

> 每个 known modulation、每个 true SNR bin 有 96 mixtures。

但整体数据规模没有交代清楚。

---

# 六、第四个重大问题：没有报告“分离质量”，这是非常大的缺口

这篇论文研究的是：

> **OOD detection from separation embeddings**

但论文几乎没有展示：

> **separator 到底把信号分离成什么样了？**

训练目标明确包含：

$$
\mathrm{SI-SDR}+\mathrm{CrossEntropy}
$$

但是正文没有给出 separation performance。

这会产生一个非常重要的 alternative explanation：

> OOD detection at chance 并不一定是因为 embedding 没有 known/unknown margin。

也可能是：

> **unknown modulation 经过 separator 后发生了严重的 representation distortion。**

因此我强烈建议增加下面这个实验：

### Experiment: Oracle vs Separated Source

比较三种输入：

| Input                          | OOD score |
| ------------------------------ | --------- |
| Clean source \(s_i\)           |           |
| Mixture \(y\)                  |           |
| Separated source \(\hat{s}_i\) |           |

然后再报告：

$$
SI\text{-}SDR(s_i,\hat{s}_i)
$$

对 known / unknown 分开统计。

这可以直接回答：

> **问题究竟来自 OOD detector，还是来自 source separation representation？**

这是目前我认为最值得补的一项实验。

---

# 七、第五个重大问题：结论中的“causal attribution”过强

论文最后说：

> “the PIT waveform-separation plus known-class cross-entropy objective shapes no known/unknown margin” 

以及：

> “because the separation objective shapes no known/unknown margin.”

这实际上是一个**机制性/因果性结论**。

但现有实验只能证明：

> 这些 post-hoc scorers 在当前 embedding 上无法有效区分 known / unknown。

不能严格证明：

> 原因就是 PIT + CE objective 导致没有 margin。

还存在其他可能：

* embedding dimension 太小/太大；
* classifier calibration 不佳；
* separation distortion；
* unknown modulation 与 known modulation 的距离确实太近；
* synthetic channel 造成表示混叠；
* score 本身不适合 complex-valued RF representation。

### 建议改成更严谨的表述：

原文：

> “the separation objective shapes no known/unknown margin”

建议改为：

> “the observed results suggest that the separation and known-class classification objectives do not provide a sufficiently exploitable known/unknown separation margin for the evaluated post-hoc OOD scorers.”

这样就从**因果结论**变成**证据支持下的解释**。

---

# 八、第六个重大问题：六种 OOD scorer 实际上没有充分展开

论文声称：

> Six post-hoc OOD scorers

包括：

Energy、MSP、ODIN、Mahalanobis、nearest-prototype、VOS-inspired。

但是 Figure 1 只展示了：

* Energy
* Prototype
* SNR-routed

而不是六个 scorer。

Table I 也把六个 scorer 压成：

> `0.433–0.524`

这种 range。

作为 reviewer，我会问：

> 为什么不直接给出六个 corrected AUROC？

因为论文的核心论点恰恰是：

> **“six scorers all fail.”**

那么六个数字就应该全部展示。

例如：

| Scorer       | −10 dB | −5 dB | ... | 20 dB | Pooled |
| ------------ | -----: | ----: | --: | ----: | -----: |
| Energy       |        |       |     |       |        |
| MSP          |        |       |     |       |        |
| ODIN         |        |       |     |       |        |
| Mahalanobis  |        |       |     |       |        |
| Prototype    |        |       |     |       |        |
| VOS-inspired |        |       |     |       |        |

哪怕放 supplementary material，正文至少应该给出完整 pooled results。

---

# 九、第七个重大问题：ODIN / Mahalanobis / VOS 的超参数是否存在隐性 leakage？

论文说：

> unknown-class data is not used for any fitted quantity. 

这是正确方向。

但是 reviewer 还会继续问：

### 谁决定了：

* ODIN temperature？
* perturbation magnitude？
* Mahalanobis covariance regularization？
* prototype construction？
* VOS parameters？
* score normalization？
* threshold？

尤其是：

> **任何 hyperparameter 如果通过看 test unknown performance 决定，都属于 leakage。**

因此建议明确增加一个：

### Hyperparameter Selection Protocol

例如：

> All scorer-specific hyperparameters were fixed using only the training/reference/validation known data and pseudo-OOD validation data. No true unknown test sample was accessed during selection.

如果某些参数是 literature default，也直接写明。

---

# 十、第八个问题：统计意义目前不够严格

论文多次强调：

> 5 seeds
> mean ± std
> significance intuition
> apparent significance

例如 artifact：

$$
0.625\pm0.031
$$

corrected：

$$
0.498\pm0.020
$$



但是从统计角度：

### 5 seeds 并不能代表测试样本不确定性。

因为这五个 seed 很可能只是：

> **同一 test set + 不同随机初始化**

也就是说：

$$
\sigma_{\text{seed}}
$$

和

$$
\sigma_{\text{test sample}}
$$

是完全不同的两个概念。

### 建议至少加入：

**95% bootstrap confidence interval**

针对 test samples 计算 AUROC CI。

更严谨一些可以同时报告：

$$
\text{mean}\pm\text{seed std}
$$

以及：

$$
95\%\ \mathrm{CI}_{bootstrap}
$$

如果比较 routed vs oracle，可以使用 paired bootstrap，因为二者使用的是完全相同的 test examples。

---

# 十一、第九个问题：校准指标报告不完整

Section IV 写得很好：

> Det@τ、FPR95、OSCR、joint accuracy alongside AUROC。

但是 Table I 实际只展示：

> Det@τ = 0.058 ± 0.038

没有真正展示：

* FPR95
* OSCR
* joint accuracy

这是一个明显的 **protocol-result mismatch**。

审稿人会直接问：

> Why introduce these metrics but not report them?

建议：

### 要么全部报告；

### 要么把 protocol 里没有用到的指标删掉。

我倾向于前者。

---

# 十二、第十个问题：global threshold 的合理性还可以进一步讨论

你们采用：

$$
\tau = P_{95}(\text{known reference scores})
$$

并且目标 FRR=5%。

这个设定符合 deployment-oriented evaluation。

但是全文自己又强调：

> OOD score scale strongly drifts with SNR.

那么马上会产生一个问题：

> **为什么一个 global threshold 在 SNR 从 −10 到 20 dB 之间是 deployment-faithful 的？**

实际上这可以变成论文一个非常有意思的额外发现。

建议同时展示：

$$
p(s|\mathrm{known},\mathrm{SNR})
$$

或者至少报告各 SNR 下：

* FRR
* detection rate
* threshold exceedance

这样可以说明：

> 即使 global threshold 极不理想，问题依旧不是 threshold selection，而是 score distribution 本身没有 OOD separation。

这个证据会让论文更加扎实。

---

# 十三、第十一个问题：unknown modulation 的覆盖范围不足

你们的 unknown：

$$
U=\{64QAM,\pi/4\text{-DQPSK},MSK,OFDM\text{-QPSK}\}
$$

这个 benchmark 可以作为一个 case study，但不能据此得出太广泛的：

> “OOD detection ... is at chance”

更准确应该限定成：

> **on the evaluated synthetic SC-BSS benchmark**

因为 open-set modulation recognition 本身已经有相当成熟的专门研究，包括利用无线信号领域知识、prototype、feature-space reconstruction 等方式处理未知 modulation。([PubMed][1])

所以你的论文不应该让 reviewer 感觉是在声称：

> “RF open-set detection fundamentally doesn't work.”

真正应该突出的是：

> **“即使一个 OOD evaluation pipeline 看起来具有非常好的 performance，只要 condition labels 没有和 score sample alignment，它仍然可能产生系统性伪结果。”**

这个定位会强很多。

---

# 十四、我还特别建议增加“far-OOD”实验

现在 unknown 都是 modulation-level unknown。

建议把 unknown 分成至少三层：

### Level 1：Near-OOD

64QAM、π/4-DQPSK 等。

### Level 2：Different modulation family

例如 FSK / APSK / OFDM variations。

### Level 3：Different waveform/domain

例如不同 pulse shaping、不同 symbol rate、不同 channel model，甚至真实 RF 数据。

这样你们最后就可以得到：

$$
\text{Near-OOD}\rightarrow
\text{Far-OOD}\rightarrow
\text{Domain-OOD}
$$

三个不同难度。

即使最后结果仍然接近 chance，也更有科学解释力。

---

# 十五、LOMO 实验不错，但解释还可以更严谨

LOMO：

> 4 splits × 3 seeds，AUROC 0.482–0.509。

这个实验很有价值，因为它证明：

> 并不是某一个特定 unknown set 导致失败。

但是：

> “boundary selected legally, on pseudo-OOD validation data only”

这里“legally”这个用词有点像在回应审稿争议，建议换掉。

可以直接写：

> “with the routing boundary selected exclusively using pseudo-OOD validation data.”

更加 scientific。

而且需要明确：

### validation data 是否完全独立于 test？

### pseudo-OOD 是否参与了 scorer hyperparameter selection？

最好做一张 split diagram：

$$
Train
\rightarrow
Known\ Validation
\rightarrow
Reference
\rightarrow
Pseudo\text{-}OOD\ Validation
\rightarrow
Unknown\ Test
$$

这样整个 leakage argument 就非常清楚。

---

# 十六、标题建议修改

现在标题：

> **A Per-Source SNR-Labeling Pitfall in Open-Set Evaluation of Single-Channel Blind Source Separation**

总体是好的。

但我更推荐：

### 版本 A

**A Per-Source SNR Labeling Pitfall in Open-Set Detection for Single-Channel Blind Source Separation**

比 `Evaluation of` 更明确你到底研究什么。

### 版本 B

**When SNR Conditioning Lies: A Label-Alignment Pitfall in Open-Set Evaluation of Single-Channel Blind Source Separation**

这个更有冲击力，但 SPL 风格我更推荐 A，比较正式。

---

# 十七、Abstract 有一个值得修改的地方

现在 abstract 有一句：

> “Because every OOD score scale drifts with noise, detection performance in this domain is meaningless without the operating condition.” 

这个表述过强。

严格说：

> **AUROC 本身并不因为 SNR 存在就“meaningless”。**

真正的问题是：

> **SNR-conditioned AUROC can become confounded when conditioning labels are misaligned.**

建议改成：

> “Because OOD score distributions can vary substantially with noise level, SNR-conditioned evaluation is particularly vulnerable to label–score misalignment.”

这个版本技术上更严谨。

---

# 十八、“standard practice”需要重新引用

你们写：

> “SNR-conditioned evaluation—per-SNR-bin AUROC profiles and pair-weighted averages—is standard practice.”

但当前给的 [10][11] 是：

* Leakage and reproducibility crisis
* Metric learning reality check

它们并不能很好支撑：

> “communication-signal OOD 中 SNR-conditioned evaluation 是 standard practice”

这是典型的 **citation mismatch**。

建议补充真正针对：

* RF OOD
* modulation open-set recognition
* AMC under SNR
* communication signal uncertainty

的文献。现在这个方向本身已有 dedicated open-set modulation recognition 工作。([PubMed][1])

---

# 十九、Figure 1 是好的，但还可以进一步增强

现在 Fig. 1 的设计其实很适合这个故事：

左边：

**Buggy labels → fake complementarity**

右边：

**Corrected labels → chance**

非常直观。

但我认为有一个更值得利用的版面：

### Figure 1 可以变成“三段式”

$$
\boxed{\text{Wrong ordering}}
\rightarrow
\boxed{\text{SNR-confounding}}
\rightarrow
\boxed{\text{Fake OOD gain}}
$$

然后下方：

$$
\boxed{\text{Correct ordering}}
\rightarrow
\boxed{\text{No confounding}}
\rightarrow
\boxed{\text{AUROC}\approx0.5}
$$

这样论文的核心 contribution 一眼就能看懂。

相比之下，现在左下角大量文字是在解释机制，占用了本来可以用于图形化展示的空间。

---

# 二十、Table I 建议重做

现在：

> `Pooled AUROC (6 scorers) 0.433–0.524`

信息量太低。

建议直接列出六个 scorer：

| Scorer       |       Buggy |   Corrected |
| ------------ | ----------: | ----------: |
| Energy       |         xxx |         xxx |
| MSP          |         xxx |         xxx |
| ODIN         |         xxx |         xxx |
| Mahalanobis  |         xxx |         xxx |
| Prototype    |         xxx |         xxx |
| VOS-inspired |         xxx |         xxx |
| Routed       | 0.625±0.031 | 0.498±0.020 |
| Oracle       |       0.681 | 0.549±0.031 |

这样 reviewer 一眼就可以看到：

> **不是某两个 scorer 出问题，而是整个 post-hoc scorer family 都没有救回来。**

---

# 二十一、我认为必须增加的 5 个实验

如果篇幅允许，我会把下面五项视为本轮大修的核心。

### ① Oracle clean-source experiment

$$
s_i
\quad vs \quad
\hat{s}_i
$$

回答：

> 是 separator 破坏了 OOD information，还是 embedding 本身没有 OOD information？

**非常重要。**

---

### ② Separation quality

报告：

$$
SI\text{-}SDR
$$

或者 SI-SDRi，在：

* SNR
* known / unknown
* kk / ku / uu

下分别报告。

---

### ③ Six scorer complete results

所有 scorer 的：

* per-SNR AUROC
* pooled AUROC
* 95% CI

不能只给 range。

---

### ④ Alignment-ablation / synthetic permutation experiment

人为构造不同 permutation：

$$
\pi_1,\pi_2,\pi_3,\ldots
$$

然后观察：

$$
\Delta AUROC
$$

这一步非常重要。

因为它可以证明：

> **这不是恰好某个 repeat/tile bug 才会出现的问题，而是更一般的 indexing mismatch phenomenon。**

这会显著提高论文的理论价值。

---

### ⑤ Far-OOD experiment

至少增加一种与当前 unknown modulation 差异明显的 OOD。

这样才能避免 reviewer 说：

> “Your negative result may simply reflect the selected unknown modulation set.”

---

# 二十二、Conclusion 建议稍微收缩

目前 Conclusion：

> “the benchmark shows post-hoc OOD detection ... is at chance” 

我建议一定加 scope：

> “Under the evaluated synthetic SC-BSS benchmark and deployment-faithful protocol, post-hoc OOD detection from separation embeddings remains close to chance...”

这样就从：

### 全球性结论

变成：

### 实验范围内的严谨结论

审稿人更容易接受。

---

# 二十三、这篇论文真正应该塑造出的 Contribution

我建议最终把 contribution 压缩成下面三个：

### Contribution 1 — Failure mechanism

发现并形式化：

> **Per-source condition/score misalignment can transform condition-dependent score drift into spurious OOD separability.**

### Contribution 2 — Diagnostic principle

证明：

> **Seed averaging, robustness perturbations, significance tests and SNR-estimation sweeps cannot reliably detect systematic label misalignment.**

### Contribution 3 — Deployment-faithful evaluation protocol

提出：

> **sequence-level alignment verification + strict train/reference/test separation + unknown-free calibration + operating-point metrics**

这三条比现在的：

> “我们发现一个 repeat/tile bug + 修正以后 AUROC=0.5”

在论文层次上高很多。

---

# 二十四、从顶级审稿人的角度，我会特别关注的“致命问题”

| 问题                                      | 严重程度         | 原因                           |
| --------------------------------------- | ------------ | ---------------------------- |
| per-source SNR 定义不清                     | **Critical** | 核心概念                         |
| 数据排列映射没有数学形式化                           | **Critical** | 核心方法学贡献                      |
| 没有 separation quality                   | **Major**    | 无法排除 alternative explanation |
| 六种 scorer 没完整报告                         | **Major**    | 核心实验不透明                      |
| hyperparameter selection 不够明确           | **Major**    | 潜在 leakage                   |
| 只有 seed std，没有 sample-level CI          | **Major**    | 统计结论不完整                      |
| corrected result scope 太宽               | **Major**    | 容易过度外推                       |
| SNR-conditioned “standard practice”引用不足 | **Major**    | citation mismatch            |
| FPR95/OSCR/joint accuracy 未报告           | **Moderate** | protocol-result 不一致          |
| Table I 信息压缩过度                          | **Moderate** | 无法验证 six-scorer claim        |
| Figure 1 很好但机制可以图形化                     | Moderate     | 可提升表达                        |
| `legal` 用词                              | Minor        | 学术表达问题                       |

---

# 二十五、如果我是 Editor，我现在会怎么判断

我的判断会是：

### **不是拒稿型论文。**

因为它有一个非常明确、容易验证、容易复现，而且对 ML-based signal processing evaluation 有现实意义的 failure mode。

尤其：

$$
0.625\rightarrow0.498
$$

这种从“漂亮结果”到“chance”的反转，本身就是很好的 scientific story。

但是目前它还有明显的：

> **“software debugging report > signal processing research paper”**

的风险。

所以大修的核心不是“再跑很多实验”，而是：

> **把代码 bug 抽象成一种可证明、可检测、可推广的 conditioned-evaluation failure mechanism。**

---

# 二十六、我认为最关键的一条修改路线

如果只给你一条最重要的修改路线，我建议把全文重新组织成：

**Problem**

$$
\text{Conditioned OOD evaluation}
$$

↓

**Failure mechanism**

$$
\pi_{\text{score}}\neq\pi_{\text{label}}
$$

↓

**Analytical explanation**

$$
\text{SNR-dependent score drift}
\Rightarrow
\text{artificial separability}
$$

↓

**Empirical demonstration**

$$
0.625\rightarrow0.498
$$

↓

**Why standard checks fail**

$$
\text{seed / robustness / SNR-error}
$$

↓

**Generalized permutation experiment**

↓

**Corrected deployment-faithful protocol**

↓

**True corrected result**

↓

**Evaluation checklist**

这样论文的主旨就不再是：

> **“我们发现了一个 bug。”**

而变成：

> **“我们证明了 conditioned OOD evaluation 中存在一种能够系统制造伪性能的标签—score alignment failure，并给出发现、验证和防范这一错误的完整方法。”**

这会明显更接近 **SPL 值得发表的 methodology letter**。

---

### 最终审稿意见

**Recommendation: Major Revision**

**优点：** 问题新颖、故事完整、错误机制清楚、corrected negative result 有价值、evaluation protocol 有潜在复用价值。

**当前主要障碍：** per-source SNR 定义、数据/index mapping、统计分析、separator quality、six-scorer 完整结果、hyperparameter leakage、OOD scope 和 causal attribution 仍不够严谨。

**我认为最值得保留的核心贡献：**

> **“systematic evaluation error can survive apparently strong statistical and robustness validation because the error is systematic rather than random.”**

这实际上比 `np.repeat` vs `np.tile` 本身更有论文价值。

另外，当前稿件已经明确写出代码、生成器、corrected dumps 和 artifact dumps均计划公开，这一点非常适合把上述 alignment audit 做成可直接运行的 reproducibility check；公开仓库本身也应固定 commit/tag，而不只是给一个 mutable branch。

如果按真正投稿版本继续修改，我建议下一步直接对这篇稿件做一次 **“逐段修改级审稿”**：我可以按照 **Abstract → Introduction → Section II → Pitfall → Corrected Results → Table I → Conclusion**，逐句指出哪些句子应该删、哪些应该弱化、哪些需要补公式/实验，并直接给出一版 **SPL 风格的修改后英文稿**。

[1]: https://pubmed.ncbi.nlm.nih.gov/37819810/?utm_source=chatgpt.com "The Importance of Expert Knowledge for Automatic Modulation Open Set Recognition."
