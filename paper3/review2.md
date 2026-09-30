我按 **IEEE Signal Processing Letters（SPL）这类期刊的正式审稿标准**看了一遍这篇稿子。总体判断是：**选题很有价值，文章的核心现象也很有意思，但目前更像一篇“有潜力的 methodological letter”，距离可以放心接收还存在几处比较关键的证据链问题。我的建议是：Major Revision（大修）**。

这不是因为文章的实验结果不好。恰恰相反，文章最有价值的地方正是它证明了一个很容易被忽视的评价管线错误，能够制造看起来非常可信的 OOD 结论：原始错误标签下，Energy/Prototype 出现明显“互补性”，路由策略 AUROC 从约 0.50 被人为抬到 \(0.625\pm0.031\)，而标签纠正后又回落到约 0.50。这个故事本身是成立且具有警示意义的。 

---

# 一、总体评价

### 1. 文章最强的地方

文章不是简单地说“代码里有一个 bug”，而是试图把它抽象成 **indexing-permutation confound**，并进一步用人工 permutation 实验说明：只要标签与真实 SNR 解耦，就可能人为制造 per-SNR OOD 结构。这个从“个别实现错误”上升到“评价方法学问题”的处理，是文章能达到 SPL 发表层次的关键。 

同时，作者没有停留在“纠正 bug 后结果没了”，而是给出了：

* verified label alignment；
* train/reference/test 严格隔离；
* unknown-free calibration；
* operating-point metrics；
* 多 scorer、多 seed、双 backbone；
* LOMO；
* near/far OOD；
* attribution/probe。

从 methodological paper 的角度，这个完整度是比较好的。 

### 2. 最大的问题

目前最大的风险不是“结论错”，而是：

> **文章部分重要结论，比实验真正能够支撑的范围更大。**

尤其是：

1. “generality”还没有被真正证明到论文表述的程度；
2. corrected result 的统计解释还不够严谨；
3. 测试数据独立性不足；
4. `kk / ku / uu` 三种混合协议没有充分拆开分析；
5. 某些数值和置信区间存在明显一致性问题；
6. “data leakage”这个措辞并不准确；
7. 最后的 attribution 对“为什么 separation destroys the OOD margin”的因果解释还不够直接。

这些问题如果不改，我认为审稿人很容易提出 **Major Revision，甚至拒稿后重投**。

---

# 二、我最关注的 Major Concerns

## Major Concern 1：文章的“general”结论目前证据还不够强

这是我认为最核心的问题。

论文写道：

> “any label–SNR decorrelation manufactures spurious per-SNR structure”

并通过 identity、repeat、reverse 和 random permutations 来支持这一点。实验确实很有说服力：identity 为 0.498，repeat 为 0.625，reverse 达到 0.765，random permutation 为 \(0.641\pm0.005\)，说明这个机制确实具有一般性。

但是，从严格的数学逻辑上说：

**“任意 label–SNR decorrelation 都会制造 spurious structure”并不是无条件成立的。**

它至少需要以下条件：

$$
p(s|c)\neq p(s|c')
$$

即 score 本身确实随 SNR 漂移；同时 permutation 必须改变不同 bin 的 SNR composition，而且路由规则或者 per-bin evaluation 要能利用这种 composition 差异。

如果：

$$
p(s|c)=p(s|c')
$$

即 score 完全不受 SNR 影响，那么再怎么打乱标签，也未必产生这种 artifact。

所以目前论文实际上证明的是：

> **When the score distribution depends on SNR, label–SNR misalignment can transform SNR-dependent score drift into spurious per-SNR OOD separability.**

这个表述比“any label–SNR decorrelation manufactures...”更加严谨。

### 建议

把“generality”改成一个**带条件的命题**，甚至可以正式写成 Proposition：

> Let \(S\) denote an OOD score whose conditional distribution varies with SNR. If the assigned SNR labels induce a different SNR composition between evaluated known and unknown bins, then the resulting conditional AUROC can deviate from the true OOD AUROC even when the score is independent of the OOD label given the true SNR.

如果能把这个写成一个 4–6 行的小定理/命题，再给一个简单推导，这篇文章的方法学高度会明显提升。

---

# 三、Major Concern 2：真正的“ODE/归因”还没有完全闭环

论文最后认为：

> 干净单源情况下存在一定 embedding margin，但在 mixture/separation 条件下消失，因此 residual interference / separation 导致 OOD margin 消失。

这部分方向是合理的，但目前只能称为 **evidence consistent with the explanation**，还不能说已经证明因果机制。

现在有三个 observation：

1. known / unknown 的 separation SI-SDR 差异很小；
2. clean single-source probe 中 Mahalanobis AUROC 可以到约 0.66；
3. separated-source 情况降到约 0.5。

但是这里缺了一组最关键的实验：

### 建议做一个“干扰强度扫描”

例如：

$$
\mathrm{SIR}\in\{-20,-15,-10,-5,0,5,10,+\infty\}\,\mathrm{dB}
$$

然后对：

* clean source；
* mixture；
* oracle separation；
* estimated separation

分别计算 Mahalanobis / Prototype AUROC。

如果得到类似：

$$
0.82\rightarrow0.75\rightarrow0.65\rightarrow0.55\rightarrow0.50
$$

这种连续下降，那么你关于“interference destroys the OOD margin”的解释就非常有说服力。

目前论文自己用了相当谨慎的：

> “Together, the observed results **suggest** ...”

这是正确的。

但如果想把文章做到更强，最好补这个控制实验。

---

# 四、Major Concern 3：`kk / ku / uu` 混合协议必须拆开报告

这一点我认为非常值得补。

现在定义：

* known pool：kk 中两个 source；
* unknown pool：ku 中 unknown slot + uu 中两个 slot。

这会引入一个潜在混杂变量：

> **一个未知 source 是否会改变另一个 source 的 separation / embedding distribution？**

例如：

### Scenario A

kk：

$$
K+K
$$

### Scenario B

ku：

$$
K+U
$$

### Scenario C

uu：

$$
U+U
$$

现在的 known pool全部来自 A，而 unknown pool同时来自 B、C。

那么即使 OOD score 没有识别 modulation novelty，它仍然可能识别：

* mixture composition；
* separation difficulty；
* interference structure；
* source assignment；
* permutation behavior。

因此现在的 pooled unknown-vs-known AUROC，实际上可能混合了：

$$
\text{OOD effect}
+
\text{mixture-composition effect}
+
\text{separation effect}.
$$

### 建议至少增加一张表

| Evaluation         | AUROC |
| ------------------ | ----: |
| K in KK vs U in KU |   ... |
| K in KK vs U in UU |   ... |
| K in KU vs U in KU |   ... |
| K in UU vs U in UU |   ... |

尤其是：

> **same-mixture comparison**

即在同一个 mixture 中比较 known slot 和 unknown slot。

这会比目前“known pool vs unknown pool”的设计更加 deployment-faithful。

---

# 五、Major Concern 4：测试集只有一个固定 seed，不能完全支撑“robustness”

论文写道：

> deterministic seed 99999

测试集固定。

而五个 seed 主要体现的是 training-run variability。

这意味着目前实际上是：

$$
\text{5 training seeds}
\times
\text{1 fixed test realization}.
$$

这与：

$$
\text{independent test-set variability}
$$

不是一回事。

bootstrap CI 能反映样本重采样的不确定性，但不能完全替代独立测试数据生成过程的 variability。

### 建议

至少增加：

$$
99999,\;100001,\;100002,\;100003
$$

这样的多个独立 test seeds，哪怕每个测试集规模减半，也比单一测试 realization 更有说服力。

尤其是你这篇文章恰恰是在研究：

> evaluation pipeline 是否会制造假象。

那么 test-set construction 的独立性本身就应该特别严格。

---

# 六、Major Concern 5：Table I 有明显的数值一致性问题

这是一个非常具体、非常容易被 reviewer 抓住的问题。

Table I 中：

* Mahalanobis corrected pooled：

$$
0.524\;[0.509,0.523]
$$

点估计 **0.524 超出了置信区间上界 0.523**。

* Prototype：

$$
0.509\;[0.494,0.507]
$$

同样：

$$
0.509 > 0.507.
$$

* VOS-inspired 同样为：

$$
0.509\;[0.494,0.507].
$$

这些数字不可能同时对应一个正常定义下的 percentile bootstrap CI。

这很可能只是一个导出/四舍五入/表格更新错误，但对于这种专门讨论 evaluation correctness 的文章而言，它会非常致命：

> **论文在证明“要严格审计评价管线”的同时，自己的统计表却存在内部不一致。**

这个必须修改，而且最好在 revision 中重新生成所有表格。

---

# 七、Major Concern 6：“at chance”这个表述稍微过头

Abstract 写：

> “is at chance across six scorers”

但正文 corrected weighted AUROC 中：

* Energy = 0.476
* MSP = 0.431
* ODIN = 0.480
* Mahalanobis = 0.520
* Prototype = 0.506
* VOS = 0.506。

所以更准确的描述应该是：

> **no practically useful OOD separation**

而不是严格意义上的：

> **all scorers are at chance**

特别是 MSP 的 0.431 已经明显偏离 0.5；当然，它是“反向区分”，但这并不等于统计意义上的 chance。

论文其实自己也承认：

> “MSP sits consistently below chance”

所以 Abstract 和 Conclusion 的措辞应该与正文一致。

### 推荐改成

> “post-hoc OOD scoring provides no reliable positive discrimination under the evaluated separation conditions”

这个表述更严谨。

---

# 八、Major Concern 7：0 dB routing boundary 的来源需要更透明

论文说：

> routing boundary (0 dB) is fixed a priori

而且 0 dB 恰好也是错误实验里 Energy / Prototype 发生互补的附近。

如果确实是完全 a priori 设定，没有问题。

但 reviewer 很自然会问：

> Why 0 dB?

尤其文章已经展示了：

* Energy 在低 SNR 更好；
* Prototype 在高 SNR 更好；
* crossover 大约就在 0 dB 附近。

因此需要明确说明：

**0 dB 是根据什么独立先验决定的？**

否则会有一种风险：

$$
\text{observe crossover}
\rightarrow
\text{choose 0 dB}
\rightarrow
\text{claim a-priori routing}.
$$

建议加入一句明确说明，例如：

> “The 0-dB boundary was fixed before any evaluation on the test set and was chosen solely from [physical prior / development protocol / training-independent criterion].”

如果确实没有完全独立的依据，那就应该老实把它定义成 development-set selected，而不是 a priori。

---

# 九、Major Concern 8：标题中的 “Per-Source SNR Label” 容易产生概念误解

标题：

> **A Per-Source SNR Labeling Pitfall...**

但正文中真正存储的是：

$$
SNR_{\mathrm{mix}}
$$

而且：

> “The per-source label stored by the evaluation pipeline is this shared SNRmix.” 

这实际上不是严格意义上的 source-specific SNR。

真实：

$$
SNR_i
=
SNR_{\mathrm{mix}}
+
10\log_{10}
\frac{\alpha_i^2}{\alpha_1^2+\alpha_2^2}.
$$

而存储的 label 是共享的 mixture SNR。

因此标题可能让人理解成：

> 每一个 separated source 有自己真实的 SNR label。

实际上不是。

### 标题可考虑

**A Label–Score Alignment Pitfall in SNR-Conditioned OOD Detection for Single-Channel Blind Source Separation**

或者：

**SNR-Label Misalignment Can Manufacture Spurious OOD Performance in Single-Channel Blind Source Separation**

后者冲击力更强，但前者更符合 SPL 风格。

---

# 十、Major Concern 9：“Data leakage”不是最准确的术语

Index Terms 最后写：

> “data leakage”

但这篇文章的核心并不是传统意义上的：

* train-test leakage；
* test-set contamination；
* label leakage；
* information leakage。

而是：

$$
\boxed{\text{label–score / label–index misalignment}}
$$

因此 “data leakage” 容易让读者误解。

建议改成：

> evaluation error / evaluation confound / label misalignment / reproducibility

比如：

**Index Terms—Blind source separation, open-set recognition, out-of-distribution detection, evaluation methodology, label misalignment.**

这会更准确。

---

# 十一、统计分析还可以进一步加强

论文已经做得不错，但对于这篇文章而言，我建议进一步强调一个问题：

### “0.50”到底意味着什么？

现在主要通过：

$$
AUROC\approx0.5
$$

来表达 failure。

但是：

$$
AUROC=0.52
$$

和

$$
AUROC=0.50
$$

并不是完全同一回事。

建议对主要 corrected results 同时报告：

$$
\Delta AUROC = AUROC-0.5
$$

以及：

* 95% CI；
* permutation test / DeLong test（如果适用）；
* effect size。

尤其是论文自己强调：

> “post-hoc oracle = 0.549 ± 0.032”

这实际上已经不是一个完全平凡的数字。作者解释说这是 per-bin max-selection 的 optimism，这是合理的。

但是最好再进一步给出：

### 一个 null oracle experiment

例如：

从六个真正独立的 chance scorers 出发，模拟相同：

* 7 SNR bins；
* 相同 sample size；
* 相同 per-bin max-selection。

展示得到的 oracle AUROC 分布。

这样你就可以非常漂亮地证明：

$$
0.549
$$

并非代表真实 OOD discrimination，而是**selection-induced optimism**。

这一项我认为会明显提升文章的方法学价值。

---

# 十二、关于“six scorers”的代表性，要稍微收敛表述

论文测试：

* Energy
* MSP
* ODIN
* Mahalanobis
* Prototype
* VOS-inspired。

这个组合已经覆盖：

$$
\text{logit-based}
+
\text{embedding-distance-based}
+
\text{synthetic virtual outlier}
$$

是合理的。

但不能因此推导：

> “post-hoc OOD detection does not work for SC-BSS”

更严谨的是：

> “the evaluated post-hoc OOD scorers do not provide reliable discrimination under the evaluated SC-BSS separation setting.”

你们其实已经在 Scope 段落里做得比较克制了：

> “these conclusions hold for the evaluated synthetic benchmark and protocol”

这一点应该保留，并在 Abstract 中也保持一致。

---

# 十三、文章的创新性到底够不够 SPL？

这是我作为 reviewer 最现实的问题。

### 如果只看：

> “我们发现自己的 repeat/tile bug。”

创新性是不够的。

但是你们真正的贡献是：

$$
\boxed{
\text{Implementation bug}
\rightarrow
\text{indexing-permutation confound}
\rightarrow
\text{general evaluation failure mode}
}
$$

并进一步证明：

$$
\text{multi-seed}
+
\text{significance}
+
\text{robustness}
$$

都可能无法发现这种系统性错误。

这一点实际上是文章最应该强化的。

目前 Introduction 中已经有这个意思：

> “multi-seed averaging is not a pipeline check”

这是非常好的观点。

我建议把整个论文的中心思想进一步浓缩成一句：

> **Randomness-oriented safeguards cannot detect deterministic pipeline errors.**

这会比“我们发现了一个 Python repeat/tile bug”更像一篇真正的方法学 letter。

---

# 十四、文章现在最值得保留的结构

我认为你现在的结构基本正确：

### ① 原错误现象

Buggy labels：

$$
AUROC_{routed}=0.625
$$

表面上非常漂亮。

### ② 形式化

$$
\sigma_q \neq \sigma_\ell
$$

把问题定义成 index permutation confound。

### ③ General permutation experiment

$$
identity\rightarrow0.498
$$

$$
repeat\rightarrow0.625
$$

$$
reverse\rightarrow0.765
$$

$$
random\rightarrow0.641
$$

这部分非常关键。

### ④ Corrected study

$$
routed=0.498\pm0.020
$$

以及各种 scorers / perturbations / LOMO / backbone。

### ⑤ Attribution

clean source 有 margin，separated source 没有。

这个 narrative 是成立的，我建议不要大改。

---

# 十五、Minor Comments

### 1. Abstract 稍微过密

Abstract 信息量非常大，一口气塞进：

* bug；
* permutation；
* routed score；
* five seeds；
* two backbones；
* attribution；
* Mahalanobis；
* checklist。

对于 SPL 来说有点过载。

建议把 Abstract 收紧成：

**Problem → artifact → formal mechanism → corrected result → methodological lesson**

五部分即可。

---

### 2. “compelling”, “dangerous”, “nobody thinks to run”等词略带评论色彩

例如：

> “The discriminating check is the one nobody thinks to run”

这种句子有传播效果，但对于 SPL，建议稍微收敛。

改成：

> “The critical check is label–score alignment, which is not captured by standard seed- or robustness-based validation.”

更学术。

---

### 3. Fig. 1 很有效，但还可以增强

目前 Fig. 1 的核心视觉表达非常清楚：

左图：

> apparent complementarity

右图：

> chance everywhere

这张图是全篇最重要的 figure。

建议增加一条非常浅的：

$$
AUROC=0.5
$$

reference line，并在 caption 中明确：

> “the same score dumps are evaluated under two different SNR-label assignments.”

这样能进一步强调：

**score 没变，只有 labels 变了。**

---

### 4. Fig. 2 很重要，但可以更数学化

现在 Fig. 2 已经能够解释：

```text
stored label:
-5 -5 -5 -5 | 10 10 10 10

true score SNR:
-5 -5 10 10 | -5 -5 10 10
```

很好。

建议在图中直接标：

$$
\sigma_q(j)
\neq
\sigma_\ell(j)
$$

这会让 Fig. 2 与正文 Section III 完全对应。

---

### 5. Table II 的 FPR95 定义需要写清楚

论文给：

$$
FPR95=0.954
$$

但建议明确写：

> FPR at 95% unknown detection rate

还是：

> FPR at 95% TPR on the OOD class

因为不同 OOD literature 中术语定义经常被读者混淆。

---

### 6. LOMO 实验建议补 boundary distribution

现在说：

> 9/12 runs choose 0 dB，3/12 找不到 crossover。

这个结果其实很有价值。

可以进一步给一个小 histogram：

$$
b^\* \in \{-5,0,5,\text{none}\}.
$$

这样更加直观地证明：

> 所谓 crossover 本身就是 instability/artifact。

---

# 十六、我认为必须修改的“硬伤”清单

如果我是正式 reviewer，我会把下面这些列为 revision 必须完成：

| 问题                                 | 严重程度   | 建议                                        |
| ---------------------------------- | ------ | ----------------------------------------- |
| Table I CI 与 point estimate 不一致    | **严重** | 重新计算全部统计量                                 |
| `any label–SNR decorrelation` 表述过强 | **严重** | 加条件、最好形式化成 proposition                    |
| kk/ku/uu 未拆分                       | **严重** | 做 same-mixture / protocol-wise evaluation |
| 单一固定 test seed                     | **严重** | 增加独立 test seeds，或明确限制结论                   |
| “at chance”表述过强                    | 中高     | 改为 “no reliable discrimination”           |
| 0 dB routing boundary 来源不透明        | 中高     | 解释独立先验来源                                  |
| attribution 因果链不足                  | 中高     | 增加 SIR/interference sweep                 |
| `data leakage`用词不准确                | 中      | 改为 label/evaluation misalignment          |
| Abstract 信息过载                      | 中      | 收紧                                        |
| Fig.2 与数学 notation 可进一步对应          | 中      | 加 \(\sigma_q,\sigma_\ell\)                |
| FPR95 定义不明确                        | 中      | 明确 metric convention                      |

---

# 十七、如果我是审稿人，我会这样写最终意见

### Recommendation: **Major Revision**

> This letter addresses an important and underappreciated failure mode in SNR-conditioned OOD evaluation for single-channel blind source separation. The central observation—that deterministic label–score misalignment can manufacture apparently strong SNR-dependent OOD performance—is technically interesting and potentially useful beyond the specific implementation error considered here. The permutation-based analysis, corrected evaluation, and attribution experiments substantially strengthen the manuscript.
>
> However, several issues currently prevent the paper from being fully convincing. In particular, the generality claim should be stated more rigorously and tied to explicit conditions on SNR-dependent score distributions; the pooled kk/ku/uu evaluation may confound modulation novelty with mixture-composition and separation effects; the use of a single fixed test realization limits the assessment of data-generation variability; and some statistical values in Table I are internally inconsistent. In addition, the claim that all corrected scorers are “at chance” should be softened or statistically substantiated, and the causal interpretation of the disappearance of embedding margins after separation would benefit from a controlled interference/SIR study.
>
> With these issues addressed, the paper could make a useful methodological contribution to reproducible evaluation of OOD detection in communication signal processing.

---

# 十八、最终判断

从“研究价值”看，我认为这篇稿子**有发表潜力**。它真正有意思的地方并不是“发现了一个代码错误”，而是：

$$
\boxed{
\text{deterministic evaluation error}
\rightarrow
\text{systematic SNR confound}
\rightarrow
\text{plausible but entirely spurious OOD story}
}
$$

尤其是你们展示了一个很值得强调的现象：

$$
\text{multi-seed}
+\text{significance}
+\text{robustness}
$$

仍然可能全部通过，而根本原因只是 label/index misalignment。这个 methodological message 是整篇论文最有价值的部分。

但是，要把它真正打造成一篇 **SPL 审稿人不容易抓住漏洞的稿子**，我建议优先解决四件事：

**第一，修正 Table I 的统计错误；第二，补 kk/ku/uu 的分组实验；第三，增强 permutation confound 的理论形式化；第四，补一个 SIR/interference-controlled attribution experiment。**

完成这四项之后，文章的论证链会从“有趣且可信的 debugging report”明显提升为“具有普适评价方法学意义的 signal-processing letter”。

