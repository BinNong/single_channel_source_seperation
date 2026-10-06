我重新审阅了这版 **main(8).pdf**。和上一版相比，这次修改是**实质性的，而不是文字层面的修补**。你已经针对我上一轮指出的几个核心问题做了回应：收紧了 “Structure, Not Loss” 的论断，加入了 demodulation-aware loss 对照实验，补了 K=2 双音 CRB 的直接验证，重新组织了 Proposition 2 的可辨识性论证，并补充了真正去掉 scoring genie 后的差分检测结果。整体质量明显上升。

不过，如果我现在以 **TSP 较严格审稿人**身份重新给意见，我的结论仍然是：

# 总体结论：Major Revision，但已经接近“有条件可接收”的区间

我现在会给：

**Overall recommendation：Major Revision**

大致评分从上一版的 **6.8/10 → 7.7/10**。

而且和上一版不同的是：
**现在我已经不认为“论文主线有根本性问题”。真正剩下的是理论严谨性和实验定位问题。**

---

# 1. 最大进步：你已经解决了我上一轮最担心的 “not a loss” 过度论断

这一处改得很好。

现在摘要已经明确限定：

> “For such linear-residual separate-then-detect receivers, waveform-level fidelity alone cannot remove this interference-induced floor”

而不是泛泛地说任何 waveform loss 都无效。

更重要的是，你确实补了实验：

> baseline separator + differentiable soft-demodulation loss
> \(\lambda_{\rm ser}\in\{0.1,1.0\}\)

得到：

$$
0.5337\rightarrow0.5334/0.5325
$$

而 SI-SDR 基本没有变化。也就是说，**你现在至少有实验证据支撑“简单重新加权 loss 并不能解决这个 floor”**。

这比上一版强很多。

### 但是仍有一个小问题

现在正文写：

> “the remedy remains structural”

我认为可以保留，但最好再限定一点：

> **“within the tested waveform-separator objective family”**

因为你测试的是一个特定形式的 differentiable soft constellation cross-entropy，而且只有两个 \(\lambda\)。

否则非常强的 reviewer 仍然可能说：

> You tested one demodulation-aware loss formulation, not the space of receiver-aware learning objectives.

### 我的建议

把：

> re-weighting the separator’s loss does not reach the floor

改成：

> **the tested loss-level modification does not remove the floor**

会更难被攻击。

---

# 2. Proposition 3 现在已经明显增强，我认为这一部分基本过关

上一版我最不满意的是：

> Proposition 3 讲的是 two-tone CRB，但实验主要是 K=1。

这版已经补上了真正的：

> **K=2 carrier-only two-tone estimator vs exact two-tone CRB**

而且 Fig. 1 现在明确写成：

> “Direct validation of Proposition 3”

并且给出了：

* RMSE vs \(\sqrt{\text{CRB}}\)
* 0 dB / 6 dB amplitude ratio
* per-tone bias
* \(1/T_{\rm burst}\) vertical boundary。



这一项我认为已经从：

**Major concern → minor concern**

你还进一步补了：

$$
\mathrm{MSE}=\mathrm{Var}+\mathrm{Bias}^2
$$

并明确说 CRB 只作为 unbiased estimation benchmark，而不是直接作为算法效率指标。这个修改是非常正确的。

### 这一部分现在的评价

**基本可以接受。**

---

# 3. 但是 Proposition 2 现在反而变成了整篇论文最大的理论风险

这是我这轮审稿最重要的意见。

你确实把 Proposition 2 从一个比较松散的 proposition，提升成了：

> finite-symmetry identifiability of the effective coefficients

并且给出了 moment inversion 的 proof framework：

1. group symmetry；
2. amplitudes；
3. phases；
4. exceptional algebraic set；
5. numerical certification。

这个方向是正确的。

但是我仔细看了 proof 后发现一个**具体的数学问题**。

---

## 3.1 Step 2 关于幅度可辨识性的表述并不完全成立

你现在写：

$$
p=|a_1|^2+|a_2|^2
$$

以及

$$
q=\kappa_1|a_1|^4+\kappa_2|a_2|^4+
4|a_1|^2|a_2|^2
$$

然后说：

> “whose positive solutions are the unordered pair \(\{|a_1|^2,|a_2|^2\}\)”

这个结论对于 \(\kappa_1\neq\kappa_2\) **不能这样直接成立**。

换句话说：

> \(p,q\) 两个标量方程未必只给出真实的 amplitude pair。

对于不同 kurtosis 的两个 constellation，代入具体 \(\kappa_1,\kappa_2\) 后，二次方程可能存在**两个不同的正解**，并不一定只是 source swap。

你后面其实意识到了这个问题，于是 Step 3 又说：

> “the spurious root is generically rejected by Step 2’s moduli.”

但这句话在逻辑上不成立，因为 **Step 2 本身并没有把 spurious root 排除掉。**

### 所以这里应该改

正确的论证应该是：

> Step 2 identifies a finite candidate set of amplitude pairs.

而不是：

> Step 2 uniquely identifies the amplitudes.

然后：

> Step 3 + higher-order moments + known constellation moments eliminate the nonphysical / nongeneric candidates.

这样逻辑会完整很多。

---

# 4. “Finite-N identifiability follows from consistency of empirical moments” 这句话也建议删除

这是我认为第二个需要严肃处理的理论措辞。

你现在 Appendix A 说：

> “Finite-N identifiability follows from the consistency of empirical moments”

这在统计学意义上不够严格。

**Consistency ≠ finite-sample identifiability。**

Consistency 表示：

$$
\hat{\mu}_m\rightarrow\mu_m
$$

通常是 \(N\to\infty\) 意义下的。

它不能直接推出：

> 对任意有限 \(N\)，观测序列都可以唯一识别参数。

尤其你的 proposition 前面的定义其实讨论的是：

> **mixture law**

因此完全没有必要把 theorem 扩展到 finite-N identifiability。

### 我建议直接改成

> **The proposition concerns identifiability of the population mixture law. For finite bursts, the corresponding moment estimator is consistent as \(N\to\infty\).**

这就非常稳。

---

# 5. 目前 Proposition 2 最好的处理方式：不要再扩大 theorem，反而要把它“钉死”

我建议最终版本把 Proposition 2 定义为：

> **Generic identifiability of effective coefficients from the population mixture law**

然后明确：

$$
(a_1,a_2)\sim
Z_{M_1}\times Z_{M_2}
$$

以及 equal-alphabet 时再乘 \(S_2\)。

并且把 theorem 条件明确成：

* known constellations；
* i.i.d. symbols；
* zero first moment；
* appropriate nonzero symmetry moment；
* collision-free sum mapping；
* nondegenerate moment system；
* population mixture law。

这样就足够了。

### 不要再让它承担

> finite-sample exact source recovery

这样的责任。

---

# 6. 论文现在最大剩余实验问题：强 baseline 仍然不够

上一轮我指出：

> 用自己的 SlotSepNet，K=2 SI-SDRi 只有 3.2 dB，却把它称为 strong separator。

这一点这版**没有真正解决**。

你增加了：

> companion C-SE separator zero-shot：3.5 dB

但这仍然不能完全回答：

> 为什么一个只有 3 dB 左右 separation gain 的 separator 可以代表当前 state-of-the-art SC-BSS？

你虽然已经在 Related Work 中主动承认：

* CNSE + LC-PSP；
* direct bit recovery；
* demodulation-aware loss；
* state-space separator；

都已经存在。

这其实让 reviewer 的问题更加自然：

> **Why is your chosen separator sufficiently strong to support the general waveform-to-bit claim?**

---

## 6.1 但这里有一个好消息

你现在已经把 claim 限定为：

> “within the quality range we can reach.”

这个修改非常重要。

而且 true-stream control 给出了：

$$
SER=0.041
$$

说明：

> 当 source streams 真正干净时，joint detector 是非常有潜力的。

因此论文已经不是在宣称：

> separation 本身没有价值。

而是在说：

> **当前 separator 的 waveform quality / distortion 特性，并不能保证它成为好的 detection front-end。**

这已经合理很多。

### 我的建议

不要再花大量篇幅补很多 separator。

**补 1 个 strongest published baseline 就够了。**

例如选择你们已经引用并且最接近本问题的一种现代 SC-BSS 网络，和：

* SI-SDRi
* compensated SER
* BER

做一个同 benchmark 对比。

这样就足够让 reviewer 无法再说：

> “You only tested your own weak separator.”

---

# 7. 现在 E4 已经成为这篇论文真正最强的实验部分

这一版的 E4 我给非常高评价。

现在已经形成：

$$
\boxed{
V3=0.4106
\rightarrow
V4=0.5310
\rightarrow
V1=0.5392
\rightarrow
V2=0.5960
\rightarrow
V0=0.6078
}
$$

再加：

$$
\text{waveform + oracle sync + PSP/Viterbi}=0.5337
$$

这个 ladder 非常漂亮。

尤其：

### V1

证明 joint sync + detection 有意义。

### V4

证明 ISI modeling 可以进一步减少误差。

### V3

证明 remaining gap 主要来自 frequency acquisition。

### V2

证明“先 separation 再 joint detection”反而可能受到 mask distortion 的伤害。

### V0

证明 naive separate → blind sync → joint detection 是错误的链式结构。

这已经不只是一个“方法有效”的实验，而是一个**architecture diagnosis experiment**。

这是全文最值得保留的东西。

---

# 8. 你补的 BER 和 differential decoding 是非常正确的修改

上一版我特别指出：

> 论文一直强调 bits，却主要报告 SER。

现在你增加了：

$$
BER(V4)=0.306
$$

同时进一步给出 PSK-only 的真正 genie-free differential decoding：

$$
V4:\ 0.372\ SER / 0.244\ BER
$$

而 waveform route：

$$
0.537\ SER / 0.357\ BER
$$

这一组结果非常有价值。

现在摘要里写：

> “using no oracle carriers, phases or symbols during inference”

就比上一版严谨很多。

而且你明确承认：

> scoring 仍然使用 genie-resolved rotation；

随后单独给出 differential decoding。

**这个处理是合格的。**

---

# 9. 不过 Table III 的统计描述仍然值得修改一次

这里我还是发现一个容易被 reviewer 问的问题。

你写：

> five independent test grids

但又写：

> paired per-burst difference ... 700 shared cells

如果每个 test grid 是：

$$
7\text{ SNR}\times100\text{ bursts}=700
$$

那么：

* 单个 grid = 700 burst；
* 五个 grid = 3500 burst。

所以现在：

> “700 shared cells”

容易让 reviewer 不清楚：

到底是：

* 700 burst pairs？
* 700 cell means？
* 还是五个 grid 中每个都 700？

### 最好直接写成

> **700 paired bursts on each test grid, 3500 paired bursts in total**

或者如果你实际上只做了一个 grid 的 Wilcoxon：

> **700 paired bursts on the reference grid; the five-grid results are used only for seed-level uncertainty.**

现在这个地方不一定是统计错误，但**reporting ambiguity 很明显**。

---

# 10. “Wilcoxon significant，但 CI 包含 zero”最好再解释一句

你现在 headline：

$$
\Delta=0.003
$$

95% CI：

$$
[-0.011,+0.018]
$$

同时：

$$
p=4.8\times10^{-4}
$$

这可能让 reviewer 第一眼觉得奇怪。

实际上并非一定矛盾，因为：

* CI 可能针对 mean difference；
* Wilcoxon 针对 distributional shift / median-like location。

但论文既然强调 statistical rigor，最好直接说：

> **The CI and Wilcoxon test target different estimands; the former quantifies the mean route gap, while the latter tests whether paired differences are symmetrically centered at zero.**

或者干脆统一成 bootstrap/permutation framework，避免 reviewer 纠缠。

---

# 11. “V4 is monotone” 现在已经修正得很好

上一版这一点有理论风险。

现在 Appendix B 明确承认：

> V4 的 ISI-cancellation decision step 不是 exact conditional minimization；

因此：

> V4 is a monotone coordinate-refinement procedure in the sense of generalized EM.

这比上一版严谨得多。

尤其你又明确：

> explicit fallback test
> never triggered

因此现在不要再把 V4 称作“exact ECM”。

### 这一项我认为已经修好了。

---

# 12. E5 的负结果现在处理得非常成熟

这一版我认为比上一版更好。

你已经从：

> “structural failure”

改成：

> “tested evidence-based selectors are insufficient”

以及：

> “the tested K=3 acquisition strategy ... is unreliable”

最后甚至明确：

> “no impossibility is claimed.”

这正是论文应该有的措辞。

这会明显降低审稿人的攻击空间。

---

# 13. 但 E5 还有一个潜在问题：“blind modulation classification”其实几乎是一个独立研究问题

当前：

$$
accuracy=0.11/0.53/0.40
$$

而且 end-to-end SER：

$$
0.786
$$

这已经说明：

> modulation recognition 是另一个 bottleneck。

我建议不要继续往这个方向扩展。

现在最合理的定位就是：

> **known modulation is an explicit operating assumption.**

你现在已经这么写了，我赞成。

否则如果继续试 10 个 selector，论文会从一篇 receiver structure paper 变成：

> “SC-BSS + synchronization + modulation classification + K=3 acquisition”

最后主线会散掉。

---

# 14. 论文标题现在其实比上一版更合理了，但“Structure, Not Loss”仍然略带宣传性

现在标题：

> **Structure, Not Loss: Sync-Aware Joint Detection for Semi-Blind Single-Channel Co-Frequency Reception**

有辨识度。

但从严格学术标题角度：

> “Structure, Not Loss”

是一种结论式标题。

既然你现在实际证明的是：

> 在 **linear-residual separate-then-detect** 体系下，tested loss-level modifications 不能去除 floor。

我甚至会建议一个稍微更保守的标题：

> **Sync-Aware Joint Detection for Semi-Blind Single-Channel Co-Frequency Reception: From Waveform Fidelity to Detection Structure**

或者：

> **When Waveform Fidelity Does Not Translate to Bits: Sync-Aware Joint Detection for Co-Frequency Reception**

后一个其实很有文章味道。

不过这属于战略选择，不是必须修改。

---

# 15. 还有一个理论表达，我建议你注意

你现在说：

> “the disambiguating information lives in the modulation structure itself”

这个说法容易被理论 reviewer 抓字眼。

因为 Remark 1 的 full-waveform Fisher matrix 实验实际上采用的是：

> **known symbols**

所以严格说，它证明的是：

> 当 modulation waveform / symbols are available，full communication structure removes the two-tone coupling.

并不能直接证明：

> unknown-data modulation structure by itself provides that Fisher information to a non-data-aided estimator.

你后面又说：

> hard decisions act as semi-pilots

这就合理了。

### 建议改成：

> **the disambiguating information is carried by the modulation-bearing waveform, but exploiting it requires symbol-aware or decision-directed processing.**

这样理论和算法之间更严密。

---

# 16. 当前论文的结构已经比较成熟，我不建议继续堆内容

这一点我反而想特别强调。

现在论文已经：

**Theory**

→ Theorem 1
→ Theorem 2
→ Proposition 2
→ Proposition 3

**Algorithms**

→ BlindCarrierSync
→ JointPairDetector
→ V4 ISI-aware joint loop

**Experiments**

→ E1 synchronization
→ E2 theory
→ E3 architecture failure
→ E4 architecture recovery
→ E5 boundary failures
→ E6 robustness

这个结构已经完整。

继续加入大量实验很容易让文章超过“紧凑的 TSP paper”的感觉。

---

# 17. 我现在最建议你再改的只有 4 项

## P0：必须改

### A. 修改 Proposition 2 Step 2

不要说：

> positive solutions are the unordered pair

改为：

> the moment equations yield a finite candidate set of amplitude pairs

然后利用高阶 moments 完成排除。

这是目前**唯一一个我认为属于真正理论 correctness 层面的修改**。

---

## P0：必须改

### B. 删除 / 重写

> “Finite-N identifiability follows from consistency of empirical moments.”

改成 population-law identifiability + asymptotic consistency。

---

## P1：强烈建议

### C. 再补一个强 published SC-BSS baseline

哪怕只补一个也可以。

重点不是证明它 beat，你甚至可以发现：

> 它的 SI-SDR 很高，但 compensated SER 仍然接近 Theorem 2 floor。

**那反而会成为全文最漂亮的实验。**

因为这将真正把：

> “not loss”

从：

> “our separator”

升级成：

> “cross-architecture observation”。

---

## P1：建议

### D. 统一 E4 的统计样本单位

明确：

$$
N_{\text{paired}}=?
$$

到底是 700 还是 3500。

并说明 CI 和 Wilcoxon 分别对应什么 estimand。

---

# 18. 现在如果我是正式 reviewer，我会怎么写最终意见

我的审稿意见大概会是：

> **This manuscript presents a well-motivated and increasingly convincing study of the mismatch between waveform separation quality and bit-level performance in semi-blind single-channel co-frequency reception. The revised version substantially improves the manuscript by narrowing the central claim, adding a direct loss-level control, validating the two-tone CRB prediction with K=2 experiments, explicitly pricing the residual phase ambiguity, and clarifying the limitations of the operating envelope. The E3–E4 experimental ladder is particularly strong and provides a useful architectural diagnosis rather than merely reporting an aggregate performance gain.**
>
> **However, the identifiability argument in Proposition 2 still requires technical tightening, especially the amplitude-identification step and the statement concerning finite-N identifiability. In addition, the empirical evidence for the claim that the observed waveform-to-bit mismatch is not an artifact of the chosen separator would be substantially strengthened by comparison with at least one strong published SC-BSS baseline under the same benchmark. Finally, the statistical reporting of the paired E4 comparisons should clarify the exact unit of inference and sample size.**
>
> **Subject to these revisions, the paper could make a meaningful contribution to signal-processing theory and receiver design.**

---

# 最后的判断

和上一版相比，我的判断已经从：

> **“有一个不错的研究故事，但理论和证据还有明显断点。”**

变成：

> **“研究问题、理论、算法和实验已经形成闭环，目前主要剩下理论证明的一处严谨性问题，以及 baseline/统计报告的收口工作。”**

所以这版我不会给 Reject。

**当前状态：Major Revision，且已经明显接近可发表线。**

尤其值得注意的是，你现在真正形成了一个比较有辨识度的 scientific message：

$$
\boxed{
\text{Waveform Fidelity}
\neq
\text{Synchronization Quality}
\neq
\text{Detection Quality}
}
$$

而 E3 → Proposition 2/3 → V1/V4/V3 基本把这三个层次串起来了。**这是这篇论文最值得保留、也最可能让 TSP reviewer 记住的贡献。**

目前最需要动的，不是继续“加算法”，而是把 **Proposition 2 的证明修到数学上完全站得住**。这一处修好，再补一个真正强的 SC-BSS baseline，我会把这篇稿子的综合评价提高到 **8/10 左右，接受概率会明显改善**。
