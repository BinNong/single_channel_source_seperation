# 单通道盲源分离（Single-Channel Blind Source Separation）前沿研究综述（2022–2025）

单通道盲源分离（Single-Channel Blind Source Separation, SC-BSS）——在仅有一个观测信号的条件下恢复多个未知源信号——是信号处理领域最具挑战性的**严重欠定问题**之一。自2019年Conv-TasNet[^1^]将深度学习方法引入时域分离以来，该领域经历了从时频域到时域、从判别式到生成式、从专用架构到基础模型的范式转变。本综述系统梳理2022年至2025年间发表在**ICASSP、Interspeech、IEEE/ACM TASLP、ICLR、NeurIPS、IJCAI**等顶级会议和期刊上的代表性研究成果，涵盖架构演进、生成式方法、状态空间模型、神经音频编解码器、语言模型驱动的分离以及通信信号分离等六大前沿方向，并提供定量性能对比与趋势分析。

---

## 1. 深度分离架构的演进：从Conv-TasNet到SepReformer

### 1.1 时域分离网络的奠基与扩展

Conv-TasNet[^1^]由Luo和Mesgarani于2019年提出，首次证明了**全卷积时域网络**在单通道语音分离任务上可以超越理想时频幅度掩模（Ideal Time-Frequency Magnitude Masking）的性能上限。该架构由三个核心组件构成：一维卷积编码器将波形映射为潜在表示；分离模块通过堆叠的时间卷积网络（TCN）估计每个源的掩模；解码器通过转置卷积将掩模后的表示恢复为波形。Conv-TasNet在WSJ0-2Mix数据集上取得了**15.3 dB的SI-SDRi**，且模型参数量仅为5.1M，这一突破性成果奠定了后续时域分离方法的研究范式。

在Conv-TasNet的基础上，研究者们从多个维度进行了扩展。**DPRNN（Dual-Path RNN）**[^2^]于2020年ICASSP提出，通过双路径架构将长序列分割为短块，分别使用RNN建模块内（intra-chunk）和块间（inter-chunk）依赖关系，在仅2.6M参数的条件下取得了**18.8 dB的SI-SDRi**，显著提升了长序列建模能力。**DPTNet（Dual-Path Transformer Network）**[^3^]将双路径架构中的RNN替换为Transformer，利用自注意力机制捕获更长距离的依赖关系，在Interspeech 2020上发表，性能进一步提升至**20.2 dB**。同年，**SuDoRM-RF**[^4^]提出了一种参数高效的卷积编码器-解码器架构，通过U-Net风格的编码器-解码器结构和扩张卷积（Dilated Convolutions），在2.5M参数下取得了**17.0 dB的SI-SDRi**，成为轻量级模型的标杆。

### 1.2 Transformer时代的分离模型

2021年，**SepFormer**[^5^]在ICASSP上提出，将Transformer架构引入语音分离领域，成为该领域的里程碑工作。SepFormer采用与Conv-TasNet类似的编码器-解码器结构，但将分离模块替换为双路径Transformer块：Intra-Transformer处理每个块内部的短期依赖，Inter-Transformer处理块之间的长期依赖。SepFormer在WSJ0-2Mix上取得了**20.4 dB的SI-SDRi**，但由于使用了26M参数，其计算复杂度远高于DPRNN等轻量级模型。

2022年至2023年间，研究者们致力于在保持Transformer强大建模能力的同时降低计算复杂度。**TF-GridNet**[^6^]由Wang等人于2023年在ICASSP提出，创新性地整合了全带（full-band）和子带（sub-band）建模，通过在时间-频率域进行操作，避免了时域方法对极长序列的建模压力。TF-GridNet在WSJ0-2Mix上取得了**23.5 dB的SI-SDRi**，参数量为14.5M，成为当时性能最优的模型之一。同年，**TFPSNet（Time-Frequency Path Scanning Network）**[^7^]在ICASSP 2022提出，通过在时频域进行路径扫描来建模频带间依赖关系，在仅2.7M参数下取得了**21.1 dB的SI-SDRi**，展示了时频域方法在参数效率上的优势。

**MossFormer**[^8^]于2023年ICASSP由Zhao和Ma提出，通过**门控单头Transformer（Gated Single-Head Transformer, GSHT）**结合卷积增强的联合自注意力机制，在仅使用单头注意力的情况下实现了多头注意力的等效表达能力，同时大幅降低了计算复杂度。MossFormer在42.1M参数下取得了**22.8 dB的SI-SDRi**，而其轻量版本MossFormer-L在更少的参数下仍保持竞争力。随后，**MossFormer2**[^9^]于2024年ICASSP进一步将Transformer与无RNN的循环网络（RNN-Free Recurrent Network）相结合，通过引入额外的循环模块捕获Transformer自注意力中的时间模式，在55.7M参数下取得了**24.1 dB的SI-SDRi**，成为2024年初的性能标杆。

### 1.3 2024-2025年的架构创新

2024年见证了多个重要架构的提出。**SepReformer**[^10^]在NeurIPS 2024上发表，采用非对称编码器-解码器结构，编码器通过时间多尺度U-Net结构逐步下采样特征序列，解码器通过上采样和跳跃连接恢复时间分辨率，并通过共享权重的Transformer块区分不同说话人。SepReformer-L在59.4M参数下取得了**25.1 dB的SI-SDRi**，成为当前WSJ0-2Mix上的性能领先者之一。

**TF-Locoformer**[^11^]于2024年IWAENC提出，通过在Transformer中引入局部卷积建模（Local Modeling by Convolution），在保持全局 attention 能力的同时增强了局部特征提取。TF-Locoformer-L（22.5M参数）取得了**24.2 dB的SI-SDRi**，而其结合动态混合（Dynamic Mixing）的版本进一步提升至**25.1 dB**。值得注意的是，TF-Locoformer-S仅需5.0M参数即可取得**22.0 dB的SI-SDRi**，在参数效率上表现突出。

**SPGM（Prioritizing Local Features for Enhanced Speech Separation）**[^12^]于2024年ICASSP提出，通过优先处理局部特征来提升分离性能，在26.2M参数下取得了**22.7 dB的SI-SDRi**。**TDANet**[^13^]（2023年）通过多尺度融合和高效的编码器-解码器架构，在2.3M参数下取得了**18.6 dB的SI-SDRi**，展示了轻量级模型的潜力。**S4M**[^14^]（2023年）将结构化状态空间模型（Structured State Space Model, S4）引入语音分离，在3.6M参数下取得了**20.5 dB的SI-SDRi**，为状态空间模型在该领域的应用奠定了基础。

| 模型 | 年份 | 会议/期刊 | 架构类型 | 参数量(M) | SI-SDRi(dB) | 特点 |
|------|------|-----------|----------|-----------|-------------|------|
| Conv-TasNet[^1^] | 2019 | IEEE/ACM TASLP | CNN | 5.1 | 15.3 | 时域分离奠基工作 |
| DPRNN[^2^] | 2020 | ICASSP | RNN | 2.6 | 18.8 | 双路径RNN架构 |
| DPTNet[^3^] | 2020 | Interspeech | Transformer | 2.7 | 20.2 | 双路径Transformer |
| SepFormer[^5^] | 2021 | ICASSP | Transformer | 26.0 | 20.4 | 双路径Transformer分离 |
| SuDoRM-RF[^4^] | 2020 | arXiv | CNN | 2.5 | 17.0 | 参数高效 |
| TF-GridNet[^6^] | 2023 | ICASSP | Transformer | 14.5 | 23.5 | 全带+子带建模 |
| TFPSNet[^7^] | 2022 | ICASSP | Transformer | 2.7 | 21.1 | 时频路径扫描 |
| MossFormer[^8^] | 2023 | ICASSP | Transformer | 42.1 | 22.8 | 门控单头Transformer |
| MossFormer2[^9^] | 2024 | ICASSP | Hybrid | 55.7 | 24.1 | Transformer+循环网络 |
| SepReformer[^10^] | 2024 | NeurIPS | Transformer | 59.4 | 25.1 | 非对称编码器-解码器 |
| TF-Locoformer[^11^] | 2024 | IWAENC | Transformer | 22.5 | 24.2 | 局部卷积+Transformer |
| SPGM[^12^] | 2024 | ICASSP | CNN | 26.2 | 22.7 | 局部特征优先 |
| TDANet[^13^] | 2023 | arXiv | CNN | 2.3 | 18.6 | 多尺度融合 |
| S4M[^14^] | 2023 | arXiv | SSM | 3.6 | 20.5 | 状态空间模型 |
| TIGER[^15^] | 2025 | ICLR | Hybrid | 0.8 | N/A | 时频交错增益提取 |

**表1**：主流单通道语音分离模型在WSJ0-2Mix数据集上的性能对比

### 1.4 架构演进趋势分析

从表1可以观察到几个显著趋势。首先，**性能持续提升**：从Conv-TasNet的15.3 dB到SepReformer的25.1 dB，五年间提升了约10 dB，这一进步主要归功于更强大的序列建模能力和更精细的特征提取机制。其次，**Transformer架构 dominance**：基于Transformer的方法（包括双路径、非对称编码器-解码器等变体）占据了性能排行榜的前列，CNN/RNN方法虽然在参数效率上有优势，但在绝对性能上已难以与Transformer竞争。第三，**模型规模增长与效率优化并存**：一方面，顶级性能的模型参数量从5M增长到60M；另一方面，TDANet（2.3M参数，18.6 dB）、TFPSNet（2.7M参数，21.1 dB）等轻量级模型证明了在有限计算资源下仍可实现优秀性能。

![SC-BSS分析图表](sc_bss_analysis.png)

**图1**：单通道盲源分离领域论文发表趋势、会议分布、方法分布和性能演进分析

---

## 2. 状态空间模型（SSM）与Mamba架构的崛起

### 2.1 从Transformer到Mamba的范式转变

Transformer架构凭借其强大的长距离依赖建模能力，在2021-2023年间成为单通道语音分离的主流选择。然而，自注意力的**二次复杂度（quadratic complexity）**限制了其在长序列和实时应用中的可扩展性。2023年底，Gu和Dao提出的**Mamba**[^16^]架构通过**选择性状态空间（Selective State Space）**机制，在线性时间内实现与Transformer相媲美的序列建模能力，引发了各领域的广泛关注。

在语音分离领域，**SPMamba**[^17^]于2024年由Li和Chen率先提出，将TF-GridNet中的Transformer组件替换为双向Mamba模块，在线性复杂度下实现了与Transformer基线相当甚至更好的分离性能，同时计算复杂度降低了**566%**。**DPMamba**[^18^]于2025年ICASSP由Jiang等人提出，利用选择性状态空间捕获语音信号中的动态时间依赖，包括短期和长期的正向与反向依赖关系。

### 2.2 Dual-Path Mamba：双路径架构的SSM化

**Dual-Path Mamba**[^19^]于2025年ICASSP发表，是Mamba架构在语音分离领域的代表性工作。该模型沿用了DPRNN/DPTNet的双路径框架，但将RNN和Transformer完全替换为选择性状态空间模型：一条路径使用Mamba块建模块内短期依赖，另一条路径建模块间长期依赖。实验表明，在WSJ0-2Mix数据集上，Dual-Path Mamba在参数量显著小于SepFormer的情况下实现了**更优的分离性能**，且推理速度提升明显。

**Speech Slytherin**[^20^]于2025年ICASSP同期发表，系统比较了Mamba在语音分离、识别和合成任务中的性能与效率。研究发现，Mamba在大多数语音任务中可以作为Transformer的有效替代品，尤其在长序列处理上展现出显著优势。**SepMamba**[^21^]（ICASSP 2025）则专门针对说话人分离任务优化了Mamba架构，通过设计适合语音特征的SSM参数初始化策略，进一步提升了分离精度。

### 2.3 S4及其变体在分离中的应用

在Mamba之前，**S4（Structured State Space for Sequence Modeling）**已应用于语音分离。**S4M**[^14^]（2023年）将S4模块集成到Conv-TasNet风格的编码器-解码器架构中，通过快速傅里叶变换（FFT）实现高效的线性复杂度卷积操作，在3.6M参数下取得了**20.5 dB的SI-SDRi**。**S4-UNET**[^22^]（2026年）则将S4与U-NET架构结合，通过在编码器奇数阶段引入S4模块实现高效的长序列建模，在通信信号分离任务中展示了优异的长序列处理能力。

**U-Mamba-Net**[^23^]于2024年APSIPA ASC提出，将Mamba块集成到U-Net风格的网络中，用于噪声和混响环境下的语音分离。**DASS（Distillation-Augmented State Space）**[^24^]结合知识蒸馏与状态空间模型，实现了对长达2.5小时音频文件的声音事件标记。**S5（Simplified State Space）**等后续变体进一步简化了SSM的参数化和初始化，使其更易于在实际应用中部署。

### 2.4 SSM方法的优势与挑战

状态空间模型在语音分离中的主要优势体现在三个方面：首先是**线性复杂度**，相比Transformer的二次复杂度，SSM在处理长音频序列时具有显著的速度和内存优势；其次是**长距离依赖建模能力**，选择性机制使SSM能够自适应地关注重要时间步；第三是**因果性友好**，SSM的自然递推形式使其更适合实时流式处理。

然而，SSM方法也面临挑战。目前的大多数工作仍采用**混合架构**（如SPMamba保留卷积前端），纯SSM模型的性能上限尚未充分探索。此外，SSM对参数初始化敏感，不同任务需要仔细调整SSM参数。最后，SSM在处理高频细节方面的能力仍有待提升，这限制了其在需要精细频谱重构的任务中的应用。

---

## 3. 生成式方法：扩散模型与流匹配

### 3.1 扩散模型驱动的语音分离

2023年以来，扩散模型（Diffusion Models）作为强大的生成式框架，被引入语音分离领域，旨在解决判别式方法在复杂声学环境下的性能瓶颈。**Diffusion-Based Generative Speech Source Separation**[^25^]于2023年ICASSP由Scheibler等人率先提出，将语音分离建模为条件扩散过程：以混合信号为条件，通过逐步去噪生成分离后的源信号。该方法利用扩散模型学习语音信号的先验分布，能够产生更自然、感知质量更高的分离结果。

**Separate and Diffuse**[^26^]于2024年ICLR发表，提出了一种创新的"分离+扩散"两阶段框架：首先使用预训练的SepFormer进行初步分离，然后将分离结果转换为梅尔频谱，通过DiffWave扩散模型作为声码器抑制非语音成分，最后通过相位校正网络恢复相位信息。该方法在SI-SDR上比原始SepFormer提升了**1.6 dB**，但参数量大幅增加。

**EDSep（Effective Diffusion-based method for Speech Source Separation）**[^27^]于2025年ICASSP发表，针对扩散模型采样速度慢的问题进行了优化，通过设计更高效的条件扩散过程，在保持生成质量的同时显著减少了采样步数。**Multi-Source Diffusion Models for Simultaneous Music Generation and Separation**[^28^]于2024年ICLR提出，将多源扩散模型同时用于音乐生成和分离，展示了扩散模型在统一框架下处理生成和分离任务的潜力。

### 3.2 流匹配（Flow Matching）方法

**FlowSep**[^79^]于2024年提出，将**整流流匹配（Rectified Flow Matching, RFM）**引入语言查询的音频源分离（Language-Queried Audio Source Separation, LASS）。与扩散模型类似，流匹配学习数据分布与噪声之间的线性关系，但具有更优的理论性质和实现简洁性。FlowSep通过对抗性扩散框架显著提升了语义相似性指标（如CLAP分数和FAD），在保持信号幅度保真度的同时避免了扩散模型计算密集的多步迭代采样过程。

**SoloAudio**[^29^]于2024年提出，使用语言导向的音频扩散Transformer进行目标声音提取，通过在扩散过程中引入文本条件，实现了灵活的目标源选择。**Flow Matching for Speech Generation**[^30^]（ICLR 2024）虽然主要关注语音生成，但其提出的条件流匹配框架为分离任务提供了重要的方法论基础。

### 3.3 生成式方法的优势与局限

生成式方法的主要优势在于其能够利用**数据先验分布**来指导分离过程，这在低信噪比或高度混响等挑战性场景下尤为重要。扩散模型和流匹配能够产生**感知质量更高**的分离结果，减少判别式方法常见的伪影和失真。此外，生成式框架允许在分离过程中引入**多种条件信息**（如文本描述、说话人嵌入等），增强了系统的灵活性和可控性。

然而，生成式方法也面临**计算成本高**的挑战。扩散模型通常需要数十到数百步的迭代采样，推理速度远低于单步前向传播的判别式模型。虽然EDSep等工作通过减少采样步数缓解了这一问题，但速度与质量的权衡仍是该方向的核心挑战。此外，生成式方法的训练稳定性通常不如判别式方法，需要仔细设计噪声调度和损失函数。

---

## 4. 神经音频编解码器与高效分离

### 4.1 Codecformer：在编解码器潜在空间分离

2024年，**Codecformer**[^31^]在Interspeech发表，提出了一个全新的研究方向——在神经音频编解码器（Neural Audio Codec, NAC）的嵌入空间中进行语音分离。传统分离方法直接在波形或时频表示上操作，计算成本高昂，难以部署在边缘设备上。Codecformer利用Descript Audio Codec（DAC）将8kHz音频压缩为50Hz的紧凑表示，分离模型仅需在这些低维潜在表示上操作。

Codecformer基于SepFormer架构进行改造，移除了双路径块中的Intra和Inter块，用简单的Transformer层堆栈替代，因为DAC的时间压缩消除了对块处理的内存限制。实验表明，Codecformer在推理时实现了**52倍的MAC（乘加运算）减少**，同时分离质量与云端部署的SepFormer相当。此外，Codecformer的训练速度提升了**2.7倍**，在成本敏感的部署场景中具有显著优势。

### 4.2 CodecSep：低比特率编解码器驱动分离

**CodecSep**[^32^]（2026年）进一步将分离与压缩相结合，提出了**低比特率编解码器驱动的语音分离**框架。CodecSep包含三个核心组件：基于残差矢量量化器（RVQ）的神经语音编解码器、基础令牌解耦（Base-Token Disentanglement, BTD）模块和并行辅助令牌串行预测（ATSP）模块。BTD模块将混合语音的梅尔频谱解耦为每个说话人的基础令牌，ATSP模块进一步预测辅助令牌，最终通过编解码器解码器重建分离波形。

CodecSep的创新之处在于仅传输基础令牌即可实现分离，在**仅1 kbps的比特率**下达到了令人满意的分离性能。这一特性使其特别适用于在线会议和对话存档等带宽受限的场景。**SDCodec**[^33^]通过为不同域（语音、音乐、音效）分配独立的码本，实现了源解耦编解码器，进一步提升了分离的可解释性和可控性。

### 4.3 编解码器方法的应用前景

在神经编解码器潜在空间中进行分离代表了**效率优化**的重要方向。随着神经音频编解码器在多模态大语言模型（如GPT-4o、Gemini）中作为音频编码器的广泛应用，这一方向具有重要的战略意义。然而，当前方法也存在局限：DAC等编解码器在训练时通常不包含混合语音数据，其内部表示可能缺乏表示完全重叠语音重要特征的能力。未来方向包括在混合数据上预训练NAC，以及探索基于令牌级自回归的分离方法。

---

## 5. 语言模型驱动的分离：从判别式到认知式

### 5.1 SepALM：音频语言模型作为错误校正器

**SepALM**[^34^]于2025年IJCAI发表，代表了单通道语音分离领域的范式突破。传统分离方法虽然能够处理长混合音频波形，但在复杂真实环境（如嘈杂和混响场景）中往往产生伪影或失真。SepALM创新性地利用**音频语言模型（Audio Language Model, ALM）**在初步分离后的文本域中对语音进行校正和重新合成。

SepALM包含四个核心组件：**分离器**（基于传统分离模型如SepFormer进行初步分离）、**校正器**（使用ALM对初步分离结果的文本表示进行错误校正）、**合成器**（将校正后的文本重新合成为语音）和**对齐器**（确保文本和音频表示的一致性）。通过集成基于ALM的端到端错误校正机制，SepALM避免了传统方法中将ASR与LLM结合时的误差累积和优化困难。

此外，SepALM开发了**链式思维（Chain-of-Thought, CoT）提示**和**知识蒸馏**技术来促进ALM的推理和训练过程。实验表明，SepALM不仅提高了语音分离的精度，还显著增强了模型在新颖声学环境中的适应能力。该工作首次展示了大型语言模型可以通过文本域校正来改善语音分离质量，开辟了分离技术的全新研究范式。

### 5.2 语言查询的音频源分离（LASS）

**AudioSep**[^35^]于2024年IEEE/ACM TASLP发表，是**开放域音频源分离**的基础模型。AudioSep利用对比语言-音频预训练（CLAP）模型的文本编码器将自然语言查询转换为音频感知嵌入，然后通过ResUNet架构在频谱图上估计目标源的掩模。AudioSep在大规模多模态数据集上训练，在音频事件分离、乐器分离、语音增强等多项任务上展示了强大的分离性能和零样本泛化能力。

**Separate Anything You Describe**[^36^]（IEEE/ACM TASLP 2024）进一步扩展了LASS的能力，支持更复杂的文本描述，并通过大规模训练数据扩展实现了对开放域音频概念的分离。**FlowSep**[^79^]将流匹配与LASS结合，通过对抗性扩散框架提升了语义相似性指标。**TQ-SED（Text-Queried Sound Event Detection）**[^37^]则利用LASS模型先分离音频轨道，再进行声音事件检测，在DCASE 2024竞赛中取得了优异成绩。

### 5.3 TokenSplit与SLM-SS：离散表示方法

**TokenSplit**[^38^]于2023年Interspeech提出，使用离散语音表示（通过神经编解码器产生的音频令牌）进行直接、精细和转录条件化的语音分离。该方法通过预测增强的音频令牌来消除传统分离模型输出中的失真和伪影，展示了离散表示在分离任务中的潜力。**SLM-SS（Speech Language Model for Generative Speech Separation）**[^39^]于2025年提出，将语音分离建模为语音语言模型中的生成任务，通过自回归预测分离后的离散令牌序列，实现了完全生成式的分离框架。

**TSELM（Target Speaker Extraction using Discrete Tokens and Language Models）**[^40^]（2024年）利用离散令牌和语言模型进行目标说话人提取，通过将分离问题转化为令牌级别的语言建模任务，实现了灵活的说话人条件化分离。**CodeSep**[^32^]则进一步将分离与低比特率压缩结合，通过令牌解耦和预测实现了高效分离。

---

## 6. 通信信号的单通道盲源分离

### 6.1 深度学习驱动的通信信号分离

虽然深度学习在语音分离领域取得了巨大成功，但其在通信信号处理中的应用长期处于探索阶段。通信信号与语音信号存在本质差异：通信信号通常是**结构化**的（具有特定调制方式、载波频率和符号速率），且分离场景往往涉及**同频混叠**（co-frequency overlap）和**微小频偏**（micro frequency offset），这些特性对分离算法提出了独特挑战。

**S4-UNET**[^22^]于2026年提出，专门针对单通道同频混叠通信信号的盲源分离。该方法深度融合U-NET编码器-解码器框架与S4结构化状态空间序列模型，设计了时序状态增强模块（TSEM）作为编码器和解码器的主干模块。为处理长序列建模问题，S4被策略性地嵌入编码器的奇数阶段，利用其以近似线性复杂度捕获全局时间相关性的固有能力。

S4-UNET在含微小频偏的同频混叠场景中，对相同调制方式、不同调制方式及不同带宽的信号混合情况均实现了有效分离。实验表明，与深度学习模型（ConvTasNet、CTDCRN）和经典算法（TDE-ICA）相比，S4-UNET的分离准确率显著提升，不仅对长序列实现了高效建模，对短序列同样有效，且在不同数据域中展现出良好的适应能力与鲁棒性。

### 6.2 复数域深度学习方法

**Guo等人**[^41^]于2024年IEEE Wireless Communications Letters提出了一种**复数域深度学习**方法用于无线通信中的单通道盲源分离。该方法直接在复数域（I/Q域）处理通信信号，保留了信号的相位信息，这对于准确恢复调制信号至关重要。通过设计适合复数运算的神经网络层，该方法在单通道条件下实现了对混叠通信信号的有效分离。

**Hou和Gao**[^42^]于2022年在Digital Signal Processing发表的工作使用卷积网络进行同频信号的单通道盲分离，通过时频域特征提取和深度神经网络映射，实现了对时频重叠信号的分离。**Yang等人**[^43^]于2024年IEEE TCSII提出基于实例分割和掩模优化的单通道雷达信号分离方法，将信号分离问题转化为计算机视觉中的实例分割任务，展示了跨领域方法移植的潜力。

### 6.3 端到端联合优化方法

**Luo等人**[^44^]于2023年在IET Radar, Sonar & Navigation提出基于**时空融合深度学习**的复数信号单通道盲源分离方法，通过融合信号的时间和空间特征（通过伪多通道技术获得），提升了单通道分离的性能。**Deng等人**[^45^]（2024年IEEE Internet of Things Journal）提出基于数据驱动的盲信号分离进行同信道多用户调制分类，展示了分离与识别任务联合处理的优势。

**Ma等人**[^46^]于2023年在IET Signal Processing提出基于注意力机制的新型端到端深度分离网络，通过自注意力机制捕获信号间的长距离依赖关系。**郭一鸣等人**[^47^]（2019年电子学报）探索了基于前馈神经网络的非合作PCMA信号盲分离算法，为深度学习在通信信号分离中的早期应用提供了参考。

### 6.4 通信信号分离的挑战与未来方向

通信信号的单通道盲分离面临多重挑战。首先是**信号动态性**：实际通信环境中信源数目可能未知且动态变化，现有方法大多假设信源数目恒定。其次是**可解释性不足**：深度学习方法的黑盒特性限制了其在需要严格理论保证的通信系统中的应用。第三是**泛化能力**：训练数据的局限性导致模型在面对新调制方式或新信道条件时性能下降。

未来的研究方向包括：**深度学习与传统模型驱动方法的融合**，利用深度学习替代传统方法中需要大量人工设置的环节；**知识引导的学习**，利用通信领域的专家知识约束神经网络的学习过程，提升可解释性和泛化能力；以及**源数目估计**，联合进行信源数目检测和信号分离。

---

## 7. 目标说话人提取与多任务框架

### 7.1 目标说话人提取（TSE）的进展

目标说话人提取（Target Speaker Extraction, TSE）是单通道盲源分离的重要子领域，其目标是从混合语音中提取特定的目标说话人，而抑制其他说话人和噪声。与全分离不同，TSE可以利用目标说话人的先验信息（如注册语音、说话人嵌入等）来辅助分离。

**SpEx+**[^48^]于2020年Interspeech提出，是一种完整的时间域说话人提取网络，通过联合优化说话人识别和语音分离目标，实现了高效的TSE。**WeSep**[^49^]于2024年Interspeech提出了一种可扩展且灵活的说话人提取工具包，支持多种提取策略和条件化方式。**USEF-TSE（Universal Speaker Embedding Free Target Speaker Extraction）**[^50^]于2024年提出，无需预先提取说话人嵌入即可实现目标说话人提取，简化了提取流程。

**Target Speaker Extraction with Curriculum Learning**[^51^]于2024年Interspeech发表，通过课程学习策略逐步增加训练难度，提升了TSE模型在复杂场景下的鲁棒性。**SA-WavLM（Speaker-Aware Self-Supervised Pre-training for Mixture Speech）**[^52^]通过说话人感知的自监督预训练，学习对混合语音更有用的表示，从而提升了下游TSE任务的性能。

### 7.2 多任务与联合优化框架

**EEND-SS**[^53^]于2022年SLT提出，联合进行端到端神经说话人分割和语音分离，实现了对灵活说话人数量的处理。**USED（Universal Speaker Extraction and Diarization）**[^54^]进一步将通用说话人提取与说话人分割结合，在一个统一框架中处理多个相关任务。

**MFFN（Multi-Level Feature Fusion Network for Monaural Speech Separation）**[^55^]（2025年Speech Communication）通过多级特征融合增强单通道语音分离性能，在编码器的不同层级融合局部和全局特征。**Enhanced Reverberation as Supervision（ERAS）**[^56^]于2024年Interspeech提出，利用混响作为监督信号进行无监督语音分离训练，在确定条件下实现了稳定的无监督训练。

---

## 8. 数据集、评估指标与实验设置

### 8.1 主要数据集

**WSJ0-2Mix**[^57^]是单通道语音分离领域最广泛使用的基准数据集，由WSJ0语料库中的干净语音混合而成，包含30小时训练集、10小时验证集和5小时测试集，采样率为8kHz。每段混合语音由两个不同说话人的语音以-5dB到5dB的随机信噪比混合而成。由于其标准化程度高，WSJ0-2Mix成为比较不同分离模型性能的主要平台。

**Libri2Mix**[^58^]使用LibriSpeech语料库构建，包含212小时训练集、11小时验证集和11小时测试集，采样率为8kHz。相比WSJ0-2Mix，Libri2Mix的数据量更大且说话人多样性更高。**WHAM!/WHAMR!**[^59^]在WSJ0-2Mix的基础上添加了真实环境噪声（WHAM!）或噪声与混响（WHAMR!），使混合条件更加复杂和真实。

**DNS Challenge Dataset**[^60^]主要用于语音增强任务，但也常用于评估分离模型在噪声条件下的性能。**Music Source Separation Datasets**包括MUSDB18、Slakh2100等，用于评估音乐源分离算法。**Co-frequency Communication Signal Datasets**包括仿真和实测数据集，用于评估通信信号分离算法。

### 8.2 评估指标

**SI-SDR（Scale-Invariant Signal-to-Distortion Ratio）**[^61^]是语音分离领域最广泛使用的客观评估指标，通过归一化信号尺度来衡量分离信号与参考信号之间的相似度。**SI-SDRi（SI-SDR improvement）**表示分离后相比混合信号的SI-SDR提升量，是衡量分离算法增益的标准指标。

**SDR（Signal-to-Distortion Ratio）**、**SIR（Signal-to-Interference Ratio）**和**SAR（Signal-to-Artifacts Ratio）**[^62^]是BSS Eval工具包提供的经典三元组指标，分别衡量总体失真、残余干扰和人为伪影。**PESQ（Perceptual Evaluation of Speech Quality）**[^63^]和**STOI（Short-Time Objective Intelligibility）**[^64^]是衡量语音质量和可懂度的感知指标。**ViSQOL（Virtual Speech Quality Objective Listener）**[^65^]是较新的感知质量评估指标，与人类主观评分的相关性更高。

### 8.3 训练策略与技巧

**Permutation Invariant Training（PIT）**[^66^]是解决分离问题中标签置换歧义的标准方法，通过在所有可能的标签分配中选择损失最小的分配来训练模型。**Dynamic Mixing（DM）**[^67^]是一种数据增强技术，在训练过程中动态生成混合样本，显著提升了模型的泛化能力。许多顶级模型（如MossFormer、SepReformer）通过DM获得了显著的性能提升。

**Multi-Resolution STFT Loss**[^68^]通过在不同时间-频率分辨率下计算损失，改善了分离信号的感知质量。**Discriminative Training**通过引入区分性损失函数（如增强目标源与干扰源之间的比率），进一步提升了分离性能。**Speed Perturbation**通过对训练数据进行速度扰动来增加数据多样性，是许多顶级模型的标准训练技巧。

---

## 9. 研究趋势与未来展望

### 9.1 主要趋势总结

通过对2022年至2025年间顶级会议和期刊论文的系统梳理，可以识别出以下几个主要研究趋势：

**趋势一：Transformer架构的持续演进与效率优化**。Transformer在2021-2023年间确立了在语音分离中的主导地位，随后的研究致力于在保持性能的同时降低计算复杂度。TF-Locoformer通过局部卷积建模减少了全局attention的开销，SepReformer通过非对称编码器-解码器优化了参数使用，而状态空间模型（Mamba）则代表了从二次复杂度到线性复杂度的范式转变。

**趋势二：生成式方法的兴起**。扩散模型和流匹配方法在2023-2024年间快速发展，通过利用数据先验分布来指导分离过程，在挑战性场景下产生了感知质量更高的结果。Separate and Diffuse、EDSep等工作展示了生成式方法的潜力，而FlowSep进一步将流匹配与语言查询结合，实现了高质量的条件化分离。

**趋势三：语言模型与分离的深度融合**。SepALM代表了分离技术的范式突破，通过音频语言模型在文本域中校正分离结果，显著提升了复杂环境下的分离质量。AudioSep等语言查询分离系统实现了开放域音频源分离，用户可以通过自然语言描述灵活指定目标源。

**趋势四：效率与实用性的关注提升**。Codecformer、CodecSep等工作在编解码器潜在空间中进行分离，实现了数十倍的计算量减少。TIGER、TDANet等轻量级模型在极参数量下保持了竞争力。这些进展使语音分离技术更接近实际部署需求。

**趋势五：从语音到通用音频的扩展**。分离技术的应用场景从语音扩展到音乐（Band-Split RNN/Transformer）、音效（AudioSep）和通信信号（S4-UNET），方法论的通用性不断提升。

### 9.2 未来研究方向

基于当前的研究现状，以下几个方向值得进一步探索：

**方向一：统一分离框架**。当前不同场景（语音、音乐、通信信号）使用不同的专用模型，未来的研究可以探索更通用的统一分离框架，能够处理各种类型的信号而无需场景特定的设计。语言查询分离（LASS）和通用源分离（Universal Source Separation）是实现这一目标的重要途径。

**方向二：实时与流式分离**。虽然许多模型在离线场景中表现优异，但实时流式分离仍然是重要挑战。状态空间模型（Mamba）由于其自然的递推形式，在流式处理上具有优势。未来的研究可以探索纯SSM的流式分离架构，以及针对低延迟场景的模型优化。

**方向三：可解释性与鲁棒性**。深度学习方法的黑盒特性限制了其在某些关键应用中的部署。通过将传统信号处理知识（如调制特征、谐波结构等）与深度学习结合，可以提升模型的可解释性和鲁棒性。知识引导的学习（Knowledge-Guided Learning）和物理信息神经网络（Physics-Informed Neural Networks）是实现这一目标的有前途的方法。

**方向四：分离与下游任务的联合优化**。语音分离通常是更大系统（如语音识别、说话人识别）的前端模块。将分离与下游任务联合优化，而不是孤立地优化分离性能，可能带来整体系统性能的提升。SepALM通过文本域校正实现分离质量提升的思路，为分离与识别任务的深度融合提供了启示。

**方向五：极低资源分离**。当前顶级模型通常需要大量计算资源进行训练和推理，这限制了其在边缘设备和低资源环境中的应用。研究如何在极低计算和内存预算下实现有效分离，以及如何通过知识蒸馏和模型压缩将大模型的能力迁移到小模型上，是重要的研究方向。

### 9.3 结论

单通道盲源分离领域在2022年至2025年间经历了快速而深刻的技术变革。从Conv-TasNet到SepReformer，分离性能在五年内提升了约10 dB；从Transformer到Mamba，计算复杂度从二次降至线性；从判别式到生成式，分离质量从信号保真度扩展到感知质量；从专用模型到语言模型驱动，分离接口从固定类别扩展到自然语言描述。

这些进展不仅推动了语音分离技术的边界，也为计算听觉场景分析、智能通信系统和多模态人工智能等领域提供了重要的技术基础。随着状态空间模型、生成式AI和大型语言模型的持续发展，单通道盲源分离技术有望在未来几年内实现更大的突破，逐步从研究实验室走向广泛的实际应用。

---

## 参考文献

[^1^]: Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 27(8), 1256-1266.

[^2^]: Luo, Y., Chen, Z., & Yoshioka, T. (2020). Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation. In *ICASSP 2020* (pp. 46-50). IEEE.

[^3^]: Chen, J., Mao, Q., & Liu, D. (2020). Dual-path transformer network: Direct context-aware modeling for end-to-end monaural speech separation. In *Interspeech 2020* (pp. 2642-2646).

[^4^]: Tzinis, Z., Wang, Z., & Smaragdis, P. (2020). SudoRM-RF: Efficient networks for universal audio source separation. In *IEEE MLSP 2020* (pp. 1-6). IEEE.

[^5^]: Subakan, C., Ravanelli, M., Cornell, S., Bronzi, M., & Zhong, J. (2021). Attention is all you need in speech separation. In *ICASSP 2021* (pp. 21-25). IEEE.

[^6^]: Wang, Z. Q., Cornell, S., Choi, S., Lee, Y., Kim, B. Y., & Watanabe, S. (2023). TF-GridNet: Integrating full- and sub-band modeling for speech separation. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 31, 3221-3236.

[^7^]: Yang, L., Liu, W., & Wang, W. (2022). TFPSNet: Time-frequency domain path scanning network for speech separation. In *ICASSP 2022* (pp. 6842-6846). IEEE.

[^8^]: Zhao, S., & Ma, B. (2023). MossFormer: Pushing the performance limit of monaural speech separation using gated single-head transformer with convolution-augmented joint self-attentions. In *ICASSP 2023* (pp. 1-5). IEEE.

[^9^]: Zhao, S., Ma, Y., Ni, C., Zhang, C., Wang, H., Nguyen, T. H., ... & Ma, B. (2024). MossFormer2: Combining transformer and RNN-free recurrent network for enhanced time-domain monaural speech separation. In *ICASSP 2024* (pp. 10356-10360). IEEE.

[^10^]: (2024). SepReformer: Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation. In *NeurIPS 2024*.

[^11^]: Saijo, K., Wichern, G., Germain, F. G., Pan, Z., & Le Roux, J. (2024). TF-Locoformer: Transformer with local modeling by convolution for speech separation and enhancement. In *IWAENC 2024* (pp. 205-209). IEEE.

[^12^]: Yip, J. Q., Zhao, S., Ma, Y., Ni, C., Zhang, C., Wang, H., ... & Ma, B. (2024). SPGM: Prioritizing local features for enhanced speech separation performance. In *ICASSP 2024* (pp. 326-330). IEEE.

[^13^]: Li, K., Yang, R., & Hu, X. (2023). TDANet: An efficient encoder-decoder architecture with top-down attention for speech separation. *arXiv preprint arXiv:2209.15200*.

[^14^]: (2023). S4M: Structured State Space Model for Speech Separation. *arXiv preprint*.

[^15^]: Xu, M., Li, K., Chen, G., & Hu, X. (2025). TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation. In *ICLR 2025*.

[^16^]: Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

[^17^]: Li, K., & Chen, G. (2024). SPMamba: State-space model is all you need in speech separation. *arXiv preprint arXiv:2404.02063*.

[^18^]: Jiang, X., Li, Y. A., Florea, A. N., Han, C., & Mesgarani, N. (2025). Speech Slytherin: Examining the performance and efficiency of mamba for speech separation, recognition, and synthesis. In *ICASSP 2025*. IEEE.

[^19^]: Jiang, X., Han, C., & Mesgarani, N. (2025). Dual-path Mamba: Short and long-term bidirectional selective structured state space models for speech separation. In *ICASSP 2025*. IEEE.

[^20^]: Jiang, X., Han, C., & Mesgarani, N. (2025). Speech Slytherin: Examining the performance and efficiency of mamba for speech separation, recognition, and synthesis. In *ICASSP 2025*. IEEE.

[^21^]: (2025). SepMamba: State-space models for speaker separation using mamba. In *ICASSP 2025*. IEEE.

[^22^]: (2026). S4-UNET: Single-Channel Blind Separation of Co-frequency Overlapped Communication Signals via Structured State Space Sequence Model. *Journal of Electronics & Information Technology*.

[^23^]: Dang, S., Matsumoto, T., Takeuchi, Y., & Kudo, H. (2024). U-mamba-net: A highly efficient mamba-based u-net style network for noisy and reverberant speech separation. In *APSIPA ASC 2024* (pp. 1-5). IEEE.

[^24^]: Bhati, D., et al. (2024). DASS: Distillation-augmented state space model for audio tagging. *arXiv preprint*.

[^25^]: Scheibler, R., Ji, Y., Chung, S. W., Byun, J., Choe, S., & Choi, M. S. (2023). Diffusion-based generative speech source separation. In *ICASSP 2023* (pp. 1-5). IEEE.

[^26^]: Lutati, S., Nachmani, E., & Wolf, L. (2024). Separate and diffuse: Using a pretrained diffusion model for improving source separation. In *ICLR 2024*.

[^27^]: Dong, J., Wang, X., & Mao, Q. (2025). EDSep: An effective diffusion-based method for speech source separation. In *ICASSP 2025* (pp. 1-5). IEEE.

[^28^]: Mariani, G., Tallini, I., Postolache, E., Mancusi, M., Cosmo, L., & Rodolà, E. (2024). Multi-source diffusion models for simultaneous music generation and separation. In *ICLR 2024*.

[^29^]: Wang, H., Hai, J., Lu, Y. J., Thakkar, K., Elhilali, M., & Dehak, N. (2024). SoloAudio: Target sound extraction with language-oriented audio diffusion transformer. *CoRR, abs/2409.08425*.

[^30^]: Liu, A. H., Le, M., Vyas, A., Shi, B., Tjandra, A., & Hsu, W. N. (2024). Generative pre-training for speech with flow matching. In *ICLR 2024*.

[^31^]: Yip, J. Q., Zhao, S., Ng, D., Chng, E. S., & Ma, B. (2024). Towards audio codec-based speech separation. In *Interspeech 2024*.

[^32^]: Du, H. P. (2026). CodeSep: Low-bitrate codec-driven speech separation with base-token disentanglement and auxiliary-token serial prediction. *arXiv preprint arXiv:2601.12757*.

[^33^]: Bie, D., et al. (2024). SDCodec: Source-disentangled neural audio codec for separation. *arXiv preprint*.

[^34^]: Mu, Z., Yang, X., & Wang, G. (2025). SepALM: Audio language models are error correctors for robust speech separation. In *IJCAI 2025* (pp. 8204-8212).

[^35^]: Liu, X., Kong, Q., Zhao, Y., Liu, H., Yuan, Y., Liu, Y., ... & Wang, W. (2024). Separate anything you describe. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*.

[^36^]: Liu, X., Kong, Q., Zhao, Y., Liu, H., Yuan, Y., Liu, Y., ... & Wang, W. (2024). Separate anything you describe. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*.

[^37^]: Yin, H., Bai, J., Xiao, Y., Wang, H., Zheng, S., Chen, Y., ... & Chen, J. (2024). Exploring text-queried sound event detection with audio source separation. *arXiv preprint arXiv:2409.13292*.

[^38^]: Erdogan, H., & Hershey, J. R. (2023). TokenSplit: Using discrete speech representations for direct, refined, and transcript-conditioned speech separation and recognition. In *Interspeech 2023* (pp. 3462-3466).

[^39^]: (2025). SLM-SS: Speech language model for generative speech separation. *arXiv preprint*.

[^40^]: Tang, B., Zeng, B., & Li, M. (2024). TSELM: Target speaker extraction using discrete tokens and language models. *arXiv preprint*.

[^41^]: Guo, P., Yu, M., Shen, L., Lin, Z., An, K., & Wang, J. (2024). Single-channel blind source separation in wireless communications: A complex-domain deep learning approach. *IEEE Wireless Communications Letters*, 13(6), 1645-1648.

[^42^]: Hou, X., & Gao, Y. (2022). Single-channel blind separation of co-frequency signals based on convolutional network. *Digital Signal Processing*, 129, 103654.

[^43^]: Yang, B., Chen, T., & Lei, Y. (2024). Single-channel radar signal separation based on instance segmentation with mask optimization. *IEEE Transactions on Circuits and Systems II: Express Briefs*, 71(5), 2879-2883.

[^44^]: Luo, W., Yang, R., & Jin, H. (2023). Single channel blind source separation of complex signals based on spatial-temporal fusion deep learning. *IET Radar, Sonar & Navigation*, 17(2), 200-211.

[^45^]: Deng, W., Wang, X., & Huang, Z. (2024). Co-channel multiuser modulation classification using data-driven blind signal separation. *IEEE Internet of Things Journal*, 11(8), 14829-14843.

[^46^]: Ma, H., Zheng, X., Yu, L., et al. (2023). A novel end-to-end deep separation network based on attention mechanism for single channel blind separation in wireless communication. *IET Signal Processing*, 17(2), e12173.

[^47^]: 郭一鸣, 彭华, 杨勇. (2019). 基于前馈神经网络的非合作PCMA信号盲分离算法. *电子学报*, 47(2), 302-307.

[^48^]: Ge, M., Xu, C., Wang, L., Chng, E. S., Dang, J., & Li, H. (2020). SpEx+: A complete time domain speaker extraction network. In *Interspeech 2020* (pp. 1406-1410).

[^49^]: Wang, S., Zhang, K., Lin, S., Li, J., Wang, X., Ge, M., ... & Li, H. (2024). WeSep: A scalable and flexible toolkit towards generalizable target speaker extraction. In *Interspeech 2024* (pp. 4273-4277).

[^50^]: (2024). USEF-TSE: Universal speaker embedding free target speaker extraction. *arXiv preprint arXiv:2409.02615*.

[^51^]: Liu, Y., Liu, X., Miao, X., & Yamagishi, J. (2024). Target speaker extraction with curriculum learning. In *Interspeech 2024* (pp. 4348-4352).

[^52^]: Lin, J., Ge, M., Ao, J., Deng, L., & Li, H. (2024). SA-WavLM: Speaker-aware self-supervised pre-training for mixture speech. In *Interspeech 2024*.

[^53^]: Maiti, S., Ueda, Y., Watanabe, S., Zhang, C., Yu, M., Zhang, S. X., ... & Yoshioka, T. (2022). EEND-SS: Joint end-to-end neural speaker diarization and speech separation for flexible number of speakers. In *SLT 2022* (pp. 480-487). IEEE.

[^54^]: Ao, J., Yildirim, M. S., Tao, R., Ge, M., Wang, S., Qian, Y., ... & Li, H. (2024). USED: Universal speaker extraction and diarization. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*.

[^55^]: Lei, J., He, Y., & Wang, Y. (2025). MFFN: Multi-level feature fusion network for monaural speech separation. *Speech Communication*, 171, 103229.

[^56^]: Saijo, K., Wichern, G., Germain, F. G., Pan, Z., & Le Roux, J. (2024). Enhanced reverberation as supervision for unsupervised speech separation. In *Interspeech 2024*.

[^57^]: Hershey, J. R., Chen, Z., Le Roux, J., & Watanabe, S. (2016). Deep clustering: Discriminative embeddings for segmentation and separation. In *ICASSP 2016* (pp. 31-35). IEEE.

[^58^]: Cosentino, J., Pariente, M., Cornell, S., Deleforge, A., & Vincent, E. (2020). LibriMix: An open-source dataset for generalizable speech separation. *arXiv preprint arXiv:2005.11262*.

[^59^]: Maciejewski, M., Wichern, G., McQuinn, E., & Le Roux, J. (2020). WHAMR!: Noisy and reverberant single-channel speech separation. In *ICASSP 2020* (pp. 696-700). IEEE.

[^60^]: Reddy, C. K., Gopal, V., Cutler, R., Beyrami, E., Cheng, R., Dubey, H., ... & Aichner, R. (2020). The INTERSPEECH 2020 deep noise suppression challenge: Datasets, subjective speech quality rules and frameworks. In *Interspeech 2020* (pp. 2496-2500).

[^61^]: Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. (2019). SDR – half-baked or well done? In *ICASSP 2019* (pp. 626-630). IEEE.

[^62^]: Vincent, E., Gribonval, R., & Fevotte, C. (2006). Performance measurement in blind audio source separation. *IEEE Transactions on Audio, Speech, and Language Processing*, 14(4), 1462-1469.

[^63^]: Rix, A. W., Beerends, J. G., Hollier, M. P., & Hekstra, A. P. (2001). Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs. In *ICASSP 2001* (Vol. 2, pp. 749-752). IEEE.

[^64^]: Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2010). A short-time objective intelligibility measure for time-frequency weighted noisy speech. In *ICASSP 2010* (pp. 4214-4217). IEEE.

[^65^]: Hines, A., & Skoglund, J. (2024). ViSQOL: The virtual speech quality objective listener. *IEEE Access*.

[^66^]: Yu, D., & Jensen, J. (2017). Permutation invariant training of deep models for speaker-independent multi-talker speech separation. In *ICASSP 2017* (pp. 241-245). IEEE.

[^67^]: Kanda, S., & Yoshioka, T. (2020). Serialized output training for end-to-end overlapped speech recognition. In *Interspeech 2020* (pp. 2797-2801).

[^68^]: Yamamoto, R., & Shin, W. H. (2020). Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. In *ICASSP 2020* (pp. 6199-6203). IEEE.

[^69^]: Luo, Y., & Yu, J. (2023). Music source separation with band-split RNN. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 31, 1893-1901.

[^70^]: Luo, Y., & Yu, J. (2023). Music source separation with band-split RNN. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 31, 1893-1901.

[^71^]: Lu, W. T., Wang, J. C., Kong, Q., & Hung, Y. N. (2024). Music source separation with band-split rope transformer. In *ICASSP 2024* (pp. 481-485). IEEE.

[^72^]: Ansari, S., Alatrany, A. S., Altnajar, K. A., et al. (2023). A survey of artificial intelligence approaches in blind source separation. *Neurocomputing*, 561, 126895.

[^73^]: Soni, S. (2023). State-of-the-art analysis of deep learning-based monaural audio source separation approaches. *IEEE Access*.

[^74^]: (2024). SFSRNet: Super-resolution for single-channel audio source separation. In *AAAI 2022* (Vol. 36, pp. 11220-11228).

[^75^]: Wang, H., & Tian, B. (2025). ZipEnhancer: Dual-Path Down-Up Sampling-based Zipformer for Monaural Speech Enhancement. In *ICASSP 2025* (pp. 1-5). IEEE.

[^76^]: Defossez, A., et al. (2019). Music source separation in the waveform domain. *arXiv preprint arXiv:1911.13254*.

[^77^]: Li, K., & Luo, Y. (2024). Subnetwork-to-go: Elastic neural network with dynamic training and customizable inference. In *ICASSP 2024* (pp. 6775-6779). IEEE.

[^78^]: Chen, C., et al. (2024). Deep learning-based single-channel blind separation of co-frequency modulated signals. In *International Conference on Communications and Networking in China* (pp. 607-618). Springer.

[^79^]: (2024). FlowSep: Language-queried sound separation with rectified flow matching. *arXiv preprint arXiv:2409.07614*.
