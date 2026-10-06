我按 **IEEE Transactions on Signal Processing（TSP）审稿人的标准**通读了这篇论文，并对关键理论、实验设计、相关工作和论文结构进行了核查。TSP 对 Regular Paper 的初投上限是 13 页，而这篇稿件正好 13 页；SPS 对审稿的核心要求包括技术正确性、贡献的清晰性、相关工作的充分性以及结果的可验证性。([IEEE信号处理学会][1])

## 一、审稿结论

**Recommendation: Reject in Present Form / Major Reconstruction Before Resubmission**

这不是说论文没有价值。相反，我认为这篇稿件有几个很好的东西：

* “**waveform quality ≠ demodulation quality**”这个问题意识是成立的，而且 E3 的设计具有一定说服力；
* QPSK/16QAM 的 separate-detection floor 分析比较漂亮；
* V1 → V4 → V3 的误差预算思想很好；
* 作者对失败案例写得相对诚实，没有刻意隐藏 K=3、调制识别、8PSK 等问题；
* 代码和合成 benchmark 声称公开，这一点对可复现性是加分项。论文明确把问题定义为比“波形恢复”更进一步的 bit-level receiver，并给出了联合检测、同步和 CRB 分析。

但是，**按照 TSP 而不是一般通信期刊的尺度，当前稿件有 4 个会直接影响录用判断的核心问题：**

> **① novelty positioning 不成立；
> ② Proposition 2 存在 identifiability 表述过度的问题；
> ③ Proposition 3 的纯双音 CRB 与真实调制通信波形之间缺乏严格桥接；
> ④ 实验并没有证明“所提出方法优于现有方法”，而主要证明了“在作者自己的 benchmark 上，某种 receiver architecture 比另一种 architecture 更适合”。**

这四点如果不重构，我作为 TSP reviewer 不会给接收意见。

---

# 二、我认为最严重的问题

## Major Concern 1：Related Work 对已有工作的描述明显过强，甚至存在事实性问题

论文在 Section II-A 中实际上把已有 learned SC-BSS 工作描述成：

> “All of these works stop at the waveform”

并以此建立本文的核心 gap。

这个表述有明显问题。

至少你论文引用的 **Hou & Gao, Digital Signal Processing, 2022** 就不是单纯停留在 waveform separation。该工作明确提出的是：

**waveform separation + LC-PSP demodulation**

并且直接报告 SER。也就是说，它已经把 separation 和 downstream demodulation 联系起来了。([科学直通车][2])

更早的 Chen et al. 2020 甚至明确提出：

> deep learning based SC-BSS can recover information bits directly

也就是直接面向 bit recovery，而不是仅仅输出 waveform。([EAI Universal Digital Library][3])

更关键的是，有一篇与你论文理论部分非常接近的 **2024 年工作**：

**“Optimal receiver and blind demodulation performance bounds for single channel co-frequency mixed signals”**

该文已经从单通道同频混合模型推导 joint ML blind detection 的充分统计量、最佳接收机以及 blind demodulation performance bound，并进一步分析信道截断误差。([杰创在线][4])

因此目前文章的叙述：

> “过去工作关注 waveform，而本文第一次把 receiver structure 引入”

这个 framing **不够成立**。

### 这会怎样影响审稿？

这是典型的 **novelty concern**。

你的真正创新可能不是：

> “从 waveform separation 转向 bit-level receiver”

因为这个方向以前已经有人做。

更可能成立的创新是：

> **在 co-frequency SC-BSS 中，系统性揭示 waveform separation、carrier synchronization 和 joint symbol detection 之间的结构性失配，并通过理论分析说明何时 separate-then-detect 会出现 error floor，以及如何利用 raw mixture 上的 joint synchronization/detection 缩小该 gap。**

这个定位就比现在准确得多。

### 必须补的 baseline

至少建议重新加入：

* PSP / Viterbi-type joint detection；
* particle filtering / Bayesian SC-BSS；
* Hou & Gao 2022 的 CNSE + LC-PSP；
* Chen et al. 的 bit-oriented DL 方法；
* 2024 optimal-receiver/performance-bound 方法；
* 2024 complex-domain DL SC-BSS；
* 2026 年最新的 C2ESDNet 类 hybrid / symbol-aware 方法；
* 2026 的 IQUMamba-1D 等长序列 SC-BSS 方法。

2026 年已经出现了进一步把 separation 与 demodulation 联合起来的 C2ESDNet-S，并专门分析 demodulation-aware loss；同时也有面向长序列 SC-BSS 的 Mamba 方法。([EurekaMag][5])

**这是当前稿件最需要补的一组实验。**

---

# 三、Major Concern 2：Proposition 2 的“carrier phase identifiability”表述存在概念混淆

论文 Section III-C 的核心说法是：

> “carrier phases are recoverable ... only through the joint discrete structure”

并提出离散 constellation 会把连续 gauge freedom 缩减为有限 rotational symmetry。

这里有一个很重要的问题：

### 你真正能 identifiably estimate 的是什么？

在实际模型中，你的未知量实际上包含：

$$
a_k = |a_k|e^{j\theta_k}
$$

也就是说，**channel phase、carrier phase、complex gain phase 本身是纠缠的。**

尤其你自己在 E4 中已经把 per-source complex gain 和 phase 一起吸收到 coupling matrix \(A\) 里面了。你的 Algorithm 事实上估计的是：

$$
A,\Delta f_1,\Delta f_2
$$

而不是独立意义上的：

$$
\phi_1,\phi_2
$$

所以严格说：

> **“carrier phase is identifiable”**

这个结论需要一个额外的 phase reference 或 channel convention。

否则：

$$
h_k e^{j\phi_k}
$$

本身就是一个复数参数，仅从观测中不能把 \(h_k\) 的相位和 \(\phi_k\) 的相位分别拆出来。

### 目前 Proposition 2 更合理的表述应该是

不是：

> carrier phase is recoverable

而应该接近：

> **the effective complex coefficient is generically identifiable up to the constellation's finite rotational symmetry and source permutation ambiguity.**

然后再说明：

> carrier phase recovery is meaningful only relative to a fixed channel-phase/reference convention.

这会严谨很多。

---

# 四、Major Concern 3：Proposition 2 → “synchronization and detection cannot be decoupled”这个推论太强

论文反复使用：

> synchronization and joint detection cannot be decoupled at \(K\ge2\)

作为 architecture-level conclusion。

但从你自己的 Proposition 3 来看，这个说法其实过强。

因为 Proposition 3 自己指出：

$$
|\Delta f|\rightarrow \text{large}
$$

时 two-tone CRB inflation 会回到 approximately 1，说明两个频率的估计逐渐独立。

那么：

> **K ≥ 2 并不自动意味着 synchronization 与 detection 必须联合。**

更准确的结论应该是：

> 在 **co-frequency / small frequency-separation / comparable-power regime** 下，independent per-source synchronization suffers from strong ambiguity and interference-induced bias.

也就是说，真正成立的是：

$$
K\ge2
$$

**不是充分条件**；

真正关键的是：

$$
|\Delta f|T_{\mathrm{burst}}\lesssim O(1)
$$

以及：

* comparable power；
* strong overlap；
* same/compatible constellation structure；
* short burst。

这是很重要的逻辑修正。

---

# 五、Major Concern 4：Proposition 3 的纯双音 CRB 与实际通信信号之间缺乏理论桥接

这是我认为**理论部分最大的硬伤**。

论文自己其实已经承认：

> “this is a pure-tone surrogate”

也就是说 Figure 3 左图的 Fisher information 是基于：

$$
s_n=\sum_k A_k e^{j(2\pi f_kn/f_s+\phi_k)}
$$

这种**未调制纯载波**模型，而不是完整的：

$$
s_n=
\sum_k h_k * [c_k[n]p[n]e^{j2\pi \Delta f_k nT_s}]
$$

的调制通信模型。

问题在于，你随后又把它升级成：

> “this explains why synchronization should go after separation”

甚至把 V4→V3 的大部分 gap 解释成 Proposition 3 所描述的 frequency-acquisition information loss。

这里缺少严格的 logical bridge。

### 为什么这是问题？

对于真实通信信号，频率偏移信息不仅存在于 carrier line，还存在于：

* modulation structure；
* cyclostationarity；
* pulse shape；
* symbol transitions；
* channel response；
* known constellation；
* possibly data-aided structure。

因此：

> pure-tone CRB ≠ communication-waveform frequency CRB.

你现在最多可以说：

> **Pure-tone Fisher analysis provides a structural diagnostic for carrier-only frequency estimation under closely spaced tones.**

但不能直接把它作为真实通信波形的 information-theoretic bound。

你自己已经写了：

> “not of the full communication waveform” 

那么后面就不能再把它当成 V4→V3 receiver gap 的严格理论下界/解释。

### 建议怎么修

至少需要增加一个真正的 communication-waveform FIM：

$$
\mathbf{\theta}
=
[
\Delta f_1,\Delta f_2,
\phi_1,\phi_2,
h_1,\ldots,h_L,
c_{1,1:N},
c_{2,1:N}
]
$$

当然完整求解可能非常复杂。

比较现实的方式是：

### 路线 A：把 Proposition 3 降级

把措辞改成：

> “carrier-only Fisher-information analysis”

明确声明它是 **diagnostic / motivation**，不是 communication receiver 的 CRB。

### 路线 B：真正建立 communication waveform CRB

即使只做：

* known symbols；
* known modulation；
* known pulse shaping；
* unknown CFO/channel；

也会比 pure-tone surrogate 强很多。

否则 TSP reviewer 很容易抓住这一点：

> **The reported CRB inflation is exact, but exact for the wrong statistical model.**

这是非常致命的一句话。

---

# 六、一个明确的数学问题：Theorem 1 中关于“任意 phase grid”的表述不成立

论文写道：

> “Averaging (4) over any grid of φ upper-bounds the corresponding phase ensemble, because the bound holds pointwise.”

这个推论不成立。

如果：

$$
P_e(\phi)\le B(\phi),\quad \forall\phi
$$

那么当然有：

$$
\int P_e(\phi)d\phi
\le
\int B(\phi)d\phi.
$$

但不能推出：

$$
\frac1G\sum_g B(\phi_g)
$$

一定大于连续 phase ensemble 的积分。

**任意 finite grid 并不保证它是连续积分的 upper bound。**

尤其你论文后面自己已经指出：

> finite phase grid 如果碰到 zero-contact rotations，会产生明显 downward bias。

所以这两个表述其实互相冲突。

正确写法应该是：

> The pointwise union bound upper-bounds the continuous phase-averaged error after integration over \(\phi\).

然后：

> a finite grid provides a numerical approximation to the phase average, whose bias depends on grid construction.

这是一个不算特别大的问题，但因为它出现在 theorem-level presentation 中，必须修。

---

# 七、Major Concern 5：你现在所谓的“blind”实际上是明显的 semi-blind

论文已经非常诚实地写了：

* known nominal carrier；
* known offset range；
* fixed timing grid；
* known modulation；
* K ∈ {1,2}；
* M-fold ambiguity 使用 genie resolution。

因此严格来说，这个 receiver 是：

> **semi-blind / modulation-conditioned / timing-assisted**

而不是通常意义上的 fully blind receiver。

尤其 E1/E4 的 SER 依赖：

> “genie M-fold rotation resolution”

这意味着真正部署时仍然存在 phase ambiguity。

论文自己说：

> differential coding in deployment

但没有真正测试 differential decoding。

因此：

### 当前实验得到的是

$$
\text{blind frequency acquisition}
+
\text{genie-assisted ambiguity resolution}
$$

而不是完全 blind end-to-end SER。

这会影响论文对“blind receiver”的所有宣传性描述。

### 建议

标题甚至可以考虑弱化：

**Structure, Not Loss: Joint Synchronization and Detection for Semi-Blind Single-Channel Co-Frequency Reception**

而不是继续强化“blind”。

---

# 八、Major Concern 6：核心实验实际上没有证明“优于 SOTA”，只证明了“raw mixture architecture 优于自己的 separator architecture”

E4 的核心结果是：

$$
V4=0.5310
$$

比：

$$
\text{waveform route}=0.5534
$$

好 2.2 percentage points。

这个结果有意义，但不能等价于：

> proposed receiver outperforms existing SC-BSS methods.

因为你的比较对象主要是：

> **作者自己的 SlotSepNet + oracle receiver**

而不是 strongest existing communication-aware receiver。

所以现在这个实验只能证明：

> **在这个 benchmark 和这个 separator 下，joint ECM on raw mixture 比 separate-then-detect 更有效。**

这其实已经是一个很不错的 paper story，但它和：

> **proposed method is state of the art**

不是一回事。

---

# 九、Major Concern 7：0.8 个百分点的 V4 gain，目前统计证据不够强

你报告：

$$
V1=0.5392
$$

$$
V4=0.5310
$$

差：

$$
0.0082
$$

即 **0.82 percentage points**。

论文给出的 seed-level 95% CI 大约：

* V1 ±0.004
* V4 ±0.005

也就是说二者的区间明显有重叠。

更关键的是，这两个方法使用的是相同 test grids，本来是可以做：

> **paired comparison**

而不应该主要依赖两个独立 CI。

应该报告：

$$
\Delta SER = SER_{V1}-SER_{V4}
$$

在每个 burst / test grid 上的 paired bootstrap CI。

例如：

* paired bootstrap 95% CI；
* Wilcoxon signed-rank；
* paired permutation test。

否则，“V4 improves V1”这个 0.8 pt 数字在 reviewer 看来可能只是 marginal。

---

# 十、Major Concern 8：E5 的样本量太小，不能承担这么强的“boundary”结论

调制识别实验：

* 每个 SNR 25 samples；
* 3 个 SNR；
* 总共 75 个样本。

最后得到：

$$
0.16/0.36/0.16
$$

pair accuracy。

正确 pair 只有：

$$
17/75
$$

然后条件 SER：

$$
0.415.
$$



这在 exploratory analysis 中可以，但如果你要用它证明：

> “blind modulation classification fails”

就明显不够。

尤其 accuracy 本身波动会非常大。

建议至少：

* 每个 SNR ≥ 1000 bursts；
* 给 confidence interval；
* confusion matrix；
* modulation-pair-wise results；
* BPSK/QPSK/8PSK/16QAM 分类概率；
* compare at least 3–5 modulation classifiers。

现在 E5 更适合称：

> **feasibility probe**

而不是 rigorous boundary experiment。

---

# 十一、一个很严重的内部矛盾：你说 separator 可以用于“counting/localisation”，但 K=2 count accuracy 只有 4.2%

论文给出的结果是：

$$
\text{count accuracy}
=
0.909/0.042/1.000
$$

对应：

$$
K=1/2/3.
$$

也就是说 K=2 时只有 **4.2%**。

而且 occupancy head 在 93% 的 K=2 mixture 上会点亮 surplus third slot。

那么结论：

> “separation serves source counting and localisation”

目前其实站不住。

更准确应该说：

> **the separator remains useful for source localisation / slot generation, although its learned count head is unreliable at K=2.**

这两个概念需要拆开：

* localisation：能够产生与真实源对应的 slot；
* counting：准确判断有几个源。

你的实验只支持前者，不支持后者。

---

# 十二、Major Concern 9：没有给出真正有说服力的 separator quality baseline

论文一直称：

> “strong learned slot separator”

但是没有系统给出：

* SI-SDR；
* SI-SDR improvement；
* SIR；
* SDR；
* PESQ 之类当然不必要；
* slot occupancy precision/recall；
* source reconstruction error；

更重要的是没有和：

* CNSE；
* complex-domain method；
* attention method；
* Mamba；
* other current SC-BSS methods

进行公平比较。

这使得 E3：

> “strong separator + oracle receiver 仍然只有 0.553”

这个结论不够扎实。

审稿人会问：

> Is the separator genuinely strong, or is this just a mediocre separator whose waveform metric does not correlate with SER?

这是必须回答的。

---

# 十三、Major Concern 10：整个 benchmark 太“干净”，导致论文结论被强烈绑定到 benchmark

你的 benchmark 固定得非常多：

* 256 symbols；
* 16 samples/symbol；
* RRC roll-off = 0.35；
* \(f_0=2\) kHz；
* offset \([-5,10]\) Hz；
* 3-tap fading；
* \(w_k\in[0.4,0.6]\)；
* SIR ≈ 0 dB；
* zero-delay timing；
* known modulation；
* K ≤ 2；
* AWGN。

这些设置非常适合做 controlled study，但不适合直接推广到：

> “SC-BSS of co-frequency communication signals” in general.

因此你最后的结论应该始终加限定：

> **under the considered short-burst, comparable-power, known-modulation, fixed-timing co-frequency regime**

否则会过度 extrapolate。

---

# 十四、尤其需要新增的泛化实验

如果我是 Reviewer #2，我会要求至少增加以下几组：

### 1. SIR sweep

不应只有：

$$
SIR=0\,dB.
$$

至少：

$$
-10,-5,0,5,10,15\,dB.
$$

你其实已经做了一部分 Fig.2，这是好的，但应该把它提升为核心实验，而不是辅助图。

### 2. Frequency separation sweep

这是全文理论核心，应该直接做：

$$
|\Delta f| =0,0.5,1,2,4,8,16\ Hz.
$$

然后画：

$$
SER,\quad f\text{-RMSE},\quad acquisition\ probability
$$

三者同时变化。

这样可以真正把 Proposition 3 → receiver performance 建起来。

### 3. Timing offset

现在 timing 被：

> “fixed by construction”

直接排除。

但这是非常强的 assumption。

至少需要：

$$
\tau \in \{0,0.1,0.25,0.5\}T_s
$$

的 sensitivity。

否则实际 receiver 可用性非常有限。

### 4. Amplitude ratio

当前：

$$
w_k\sim U(0.4,0.6)
$$

太窄。

应该测试例如：

$$
-20\sim20\,dB
$$

甚至至少：

$$
-10,-5,0,5,10\,dB.
$$

---

# 十五、Major Concern 11：V3 不应该叫 “oracle-carrier bound”

Table III 中：

$$
V3=0.4106
$$

被当成：

> oracle-carrier bound / ceiling

但实际上 V3 仍然：

* 需要估计 gains；
* 需要估计 symbols；
* 使用有限 L=5 model；
* 仍然存在 channel mismatch。

所以它不是 fundamental bound。

更严谨的名称应该是：

> **oracle-frequency reference**

或者：

> **oracle-carrier-offset reference**

真正的理论 lower bound 应来自：

* perfect frequency；
* perfect channel;
* exact likelihood；
* exact MAP/ML sequence detection。

目前你把：

$$
0.4106
$$

称为某种 “bound”，容易被 reviewer 抓。

---

# 十六、Major Concern 12：V4 的 1.6 s/burst 很难被称为实用 receiver

论文给出：

> V1+V4 ≈ 1.6 s / burst / one CPU core

而 burst duration 是：

$$
0.256s.
$$

也就是说：

$$
\frac{1.6}{0.256}\approx6.25
$$

即单核处理时间约为 burst 时长的 6.25 倍。

论文自己也承认它主要是：

> offline-analysis cost

但你的论文定位是 receiver architecture，这个问题不能完全避开。论文方法包含约 750 次 converged classification-EM invocation，且 frequency grid 与 joint hypothesis 随规模快速增长。

至少需要：

* CPU vs GPU；
* latency；
* throughput；
* memory；
* complexity in \(N\)；
* complexity in \(M_1M_2\)；
* complexity in frequency grid size。

最好给一个和 PSP / LC-PSP / neural receiver 的 complexity table。

---

# 十七、理论上还有一个很值得加强的地方：你其实有一个更漂亮的核心理论，可以把论文重新“立住”

目前论文的理论部分分散成：

1. joint detection union bound；
2. separate detection floor；
3. phase identifiability；
4. two-tone CRB inflation。

看起来像四个相对独立的结果。

实际上可以把它重新组织成一个非常漂亮的主线：

$$
\boxed{
\text{Waveform Separation}
\rightarrow
\text{Synchronization}
\rightarrow
\text{Detection}
}
$$

对应三个不同的信息损失：

### Layer 1：waveform metric gap

$$
SI\!-\!SDR \not\Rightarrow SER
$$

### Layer 2：synchronization ambiguity

$$
\Delta f_1,\Delta f_2
$$

在 overlapping regime 下不可独立估计/强烈耦合。

### Layer 3：symbol detection geometry

$$
d_{\min}
\left(
\mathcal C_1+ae^{j\phi}\mathcal C_2
\right)
$$

决定 joint detector 的 SER，而 separate detector 出现 exact floor。

这样全文就可以围绕一个 theorem-like statement：

> **For short-burst comparable-power co-frequency mixtures, waveform-fidelity optimization, independent synchronization, and per-source detection optimize different statistical objects; therefore a high-SI-SDR separator need not be an effective detection front-end.**

这个定位比现在单纯堆四个 theoretical findings 强得多。

---

# 十八、E3 是这篇论文真正值得保留的核心实验

我反而认为 E3 是全文最有论文价值的部分。

你做了：

$$
\text{separation}
\rightarrow
\text{oracle sync}
\rightarrow
\text{detection}
$$

得到：

$$
K=1:0.1454
$$

而：

$$
K=2:0.5534
$$

以及：

$$
K=3:0.5858.
$$

然后把 oracle sync 换成 blind sync，性能继续恶化，并最终发现：

$$
\text{separate}\rightarrow\text{blind sync}
\approx
\text{raw mixture}\rightarrow\text{blind sync}.
$$

这部分的逻辑是清楚的。论文也明确把 E3 定义成“waveform-to-bit gap”的直接实验。

**这个实验其实足以成为文章主线。**

建议把大量篇幅从：

> 纯数学细节 + E5 小样本 negative result

挪到：

> **why SI-SDR-optimal separation can be SER-suboptimal**

上面。

---

# 十九、E4 的 V1 → V4 → V3 ladder 是第二个亮点

这一部分也设计得不错：

$$
V1=0.5392
$$

$$
V4=0.5310
$$

$$
V3=0.4106
$$

对应：

$$
\text{memoryless model}
\rightarrow
\text{ISI-aware model}
\rightarrow
\text{oracle frequency}
$$

这实际上形成了一个很清楚的 error budget：

$$
\underbrace{0.5392-0.5310}_{\text{model mismatch}}
+
\underbrace{0.5310-0.4106}_{\text{frequency acquisition}}
$$

这比单纯说“我们的方法 SER 更低”强很多。论文已经在这个方向上做了很好的铺垫。

但是需要把“bound”“ceiling”“first pipeline”等用词降级。

---

# 二十、Minor Comments

这些不会单独导致拒稿，但建议全部修。

### 1. Abstract 太密

第一段塞入大量：

* SER；
* CRB；
* E3；
* E4；
* E5；
* K=3；
* modulation classifier。

信息量过高。

顶刊摘要应该突出：

> problem → insight → method → main result

而不是几乎把全文 numerical results 都塞进去。

---

### 2. “first pipeline”必须删除或极度谨慎

你现在写：

> “To our knowledge, this is the first pipeline...”

在已有：

* 2020 bit-level DL；
* 2022 separation-demodulation；
* 2024 optimal receiver；
* 2024 modulation-classification + separation；
* 2026 symbol-aware hybrid methods

的情况下，这句话风险非常大。([科学直通车][2])

建议改：

> “To our knowledge, this is the first study on this benchmark to explicitly quantify ...”

这种表述安全很多。

---

### 3. “exact CRB”也需要限定

应该写：

> exact two-tone pure-carrier CRB under the specified nuisance model

而不是让人理解成：

> exact CRB of the communication receiver.

---

### 4. Figure 3 理论假设需要更醒目

建议在 caption 或正文里明确：

> **carrier-only surrogate, not full communication waveform**

这一点现在虽然有，但容易被读者忽略。

---

### 5. Figure 4 很漂亮，但应该加入 constellation collision 的 analytic explanation

这一图其实是全文最容易让读者理解理论的图之一。

最好增加：

$$
d_{\min}(\phi)=0
$$

对应的 exact collision examples。

---

### 6. Figure 1 / Table II 的信息密度偏高

视觉上不算差，但第 8–10 页的信息密度明显偏高，尤其 Table III + Algorithm 1 + discussion 被压在一个很小的版面里。

从版面上看，论文没有超出 TSP 初投 13 页限制，但确实已经接近信息密度上限。([IEEE信号处理学会][1])

---

# 二十一、如果我是 TSP Reviewer，我会写出的正式评审结论

可以概括为：

> **The manuscript presents an interesting and potentially important observation that waveform-fidelity-based single-channel source separation does not necessarily translate into communication-level detection performance. The experimental decomposition between separate detection, blind synchronization, joint detection, and ISI-aware modeling is thoughtful, and the exact separate-detection floor for QPSK/16QAM is potentially useful.**
>
> **However, the manuscript in its current form has substantial issues in novelty positioning, theoretical interpretation, and experimental benchmarking. In particular, the related-work discussion overstates the absence of receiver-aware SC-BSS methods; prior works have already addressed joint separation/demodulation and bit-level recovery, and a 2024 study directly derives optimal reception and blind-demodulation performance bounds for single-channel co-frequency mixtures. Moreover, the carrier-phase identifiability claim is not cleanly separated from the unknown complex channel phase, and the pure-tone two-tone CRB is not a rigorous CRB for the actual modulated, pulse-shaped, faded communication waveform. Finally, the main numerical advantage of V4 over V1 is modest, while the paper lacks sufficiently strong comparisons against classical optimal/near-optimal SC-BSS receivers and recent communication-aware deep learning methods.**
>
> **I therefore cannot recommend acceptance in the present form. A substantially revised manuscript with a sharper novelty claim, corrected identifiability and CRB statements, stronger baselines, paired statistical testing, and broader robustness experiments could become a valuable contribution.**

---

# 二十二、我的最终判断

从“有没有研究价值”来看：

**有。**

从“有没有值得继续做的核心 idea”来看：

**有，而且 E3 + V1/V4/V3 error-budget 这条线是有潜力的。**

但从 **TSP 当前版本是否达到录用标准**来看：

> **目前还不够。**

最关键的不是“再增加几个实验”，而是要进行一次 **scientific repositioning**：

$$
\boxed{
\text{从“我提出了一个联合接收算法”
}
\rightarrow
\text{“我揭示并证明了 SC-BSS 中 waveform metric、synchronization 和 detection 之间的结构性失配，并提出相应的 receiver architecture”
}
}
$$

然后围绕这个核心重新处理：

$$
\boxed{
\text{Novelty}
\rightarrow
\text{Identifiability}
\rightarrow
\text{CRB rigor}
\rightarrow
\text{Fair baseline}
\rightarrow
\text{Statistical validation}
}
$$

这是我认为这篇稿件从“容易被 Reject”走向“具备 TSP 竞争力”最关键的一轮修改方向。

[1]: https://signalprocessingsociety.org/publications-resources/guidelines-associate-editors?utm_source=chatgpt.com "Guidelines for Associate Editors | IEEE Signal Processing Society"
[2]: https://www.sciencedirect.com/science/article/abs/pii/S1051200422002718?utm_source=chatgpt.com "Single-channel blind separation of co-frequency signals based on convolutional network - ScienceDirect"
[3]: https://eudl.eu/doi/10.1007/978-3-030-41114-5_45?utm_source=chatgpt.com "Deep Learning Based Single-Channel Blind Separation of Co-frequency Modulated Signals - EUDL"
[4]: https://www.joconline.com.cn/zh/article/doi/10.11959/j.issn.1000-436x.2024265/?utm_source=chatgpt.com "单通道同频混合信号的最佳接收及盲解调性能界"
[5]: https://eurekamag.com/research/103/329/103329705.php?utm_source=chatgpt.com "Single-Channel Blind Source Separation of Co-Channel Communication Signals: A Hybrid Knowledge-Data Driven Approach"
