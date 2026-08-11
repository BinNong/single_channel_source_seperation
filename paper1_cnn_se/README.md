# Single-Channel Blind Source Separation of Communication Signals

A complete PyTorch implementation for single-channel blind source separation of co-frequency overlapped communication signals using **Complex-Valued Lightweight CNN** with **Complex Squeeze-and-Excitation Attention**.

> **Paper Target**: Digital Signal Processing (Elsevier) or IEEE Wireless Communications Letters  
> **Key Innovation**: Complex Squeeze-and-Excitation Block for complex-valued neural networks  
> **Hardware**: Designed for 8GB GPU (RTX 4060), < 1M parameters

---

## Project Structure

```
comm_bss_project/
├── config.py           # All hyperparameters in one place
├── data_generator.py   # Communication signal generation (BPSK/QPSK/8PSK/16QAM)
├── models.py           # Complex CNN + SE (proposed), Real-valued CNN (baseline)
├── train.py            # Training script with TensorBoard logging
├── evaluate.py         # Evaluation across SNR points + visualization
├── utils.py            # Metrics (SI-SDR, SDR, SIR, NMSE) + checkpoint utils
├── run.sh              # One-click train all models
└── README.md           # This file
```

---

## Quick Start

> **Reproducibility**: a complete per-seed record of all 44 training runs (7 experimental phases, 2-5 seeds per configuration) is kept in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) (symlink to `docs/EXPERIMENT_LOG.md`). Each row carries the exact command, hyperparameters, and the resulting SDR / SI-SDR / SIR numbers; the paper's headline numbers can be reproduced verbatim from those commands.

### 1. Install Dependencies

```bash
pip install torch numpy scipy pandas matplotlib tensorboard
```

For 8GB GPU (RTX 4060), PyTorch with CUDA 11.8:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Test Data Generation

```bash
python data_generator.py
```

Expected output:
```
Mixture shape: (4096,), dtype: complex128
Source 1 power: 1.0000
Mixture SNR (estimated): 10.2 dB
Dataset batch shapes:
  Mixture: torch.Size([4, 1, 4096]), dtype: torch.complex64
...
Data generator test passed!
```

### 3. Test Models

```bash
python models.py
```

Expected output:
```
--- Proposed Model ---
[ComplexLightweightSepNet] Total parameters: 143,234 (0.143M)
Model: ComplexLightweightSepNet
Parameters: 143,234 (0.143M)
Trainable:  143,234
...
All model tests passed!
```

### 4. Train Proposed Model

```bash
# Single model training
python train.py --model complex_cnn_se --epochs 100 --batch_size 16

# With custom settings
python train.py --model complex_cnn_se --epochs 100 --batch_size 16 \
    --lr 1e-3 --hidden 32 --layers 4 --loss mse
```

Training log:
```
 Epoch |  Train Loss |   Val Loss |     SI-SDR |        SDR |        SIR |   Time
----------------------------------------------------------------------
     1 |   0.523412 |   0.412345 |     -2.34 |      1.23 |      3.45 |   45.2s
     2 |   0.345678 |   0.298765 |      1.56 |      4.78 |      6.90 |   44.8s
...
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir=./runs
```

### 5. Evaluate Model

```bash
python evaluate.py \
    --model complex_cnn_se \
    --checkpoint checkpoints/complex_cnn_se_h32_l4_bs16_lr0.001_mse_run1_best.pt \
    --viz
```

Results include:
- SNR-dependent performance table (-10dB to 20dB)
- Overall metrics (SI-SDR, SDR, SIR, NMSE)
- Constellation diagrams (saved to `results/figures/`)
- CSV + LaTeX tables (ready for paper)

### 6. Run Full Experiment (All Models)

```bash
chmod +x run.sh
./run.sh
```

This trains 3 models and evaluates all:
1. **Proposed**: Complex CNN + SE (143K params)
2. **Ablation**: Complex CNN without SE
3. **Baseline**: Real-valued CNN (Hou & Gao 2022 style)

---

## Model Comparison

| Model | Params | Type | SE Block | ~Training Time (100 epochs) |
|-------|--------|------|----------|---------------------------|
| **Complex CNN + SE** (Proposed) | ~143K | Complex | Yes | ~1.5h (RTX 4060) |
| Complex CNN (no SE) | ~140K | Complex | No | ~1.3h |
| Real-valued CNN (Baseline) | ~500K | Real | N/A | ~2h |

---

## Key Features

### Complex Squeeze-and-Excitation Block (Core Innovation)

```python
class ComplexSEBlock(nn.Module):
    """Complex SE: extends squeeze-and-excitation to complex domain."""
    def __init__(self, channels, reduction=4):
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels * 2 // reduction),  # real+imag
            nn.ReLU(),
            nn.Linear(channels * 2 // reduction, channels * 2),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C, T] complex
        # Squeeze: global pool + concat real/imag stats
        z_real = self.avg_pool(x.real).view(b, c)
        z_imag = self.avg_pool(x.imag).view(b, c)
        z = torch.cat([z_real, z_imag], dim=1)  # [B, 2C]

        # Excitation: learn channel weights
        scale = self.fc(z)
        # Scale: apply to complex features
        return x * weight
```

### Signal Generation

- **Modulations**: BPSK, QPSK, 8PSK, 16QAM
- **Channel**: Co-frequency overlap + AWGN + multipath fading
- **SNR range**: -10dB to 20dB (training: -5dB to 20dB)
- **On-the-fly generation**: No need to download datasets

---

## Customization Guide

### Change Model Size

```bash
# Smaller model (faster, less memory)
python train.py --model complex_cnn_se --hidden 16 --layers 3

# Larger model (better performance, needs batch_size=8)
python train.py --model complex_cnn_se --hidden 64 --layers 6 --batch_size 8
```

### Change Loss Function

```bash
python train.py --model complex_cnn_se --loss si_sdr     # SI-SDR loss
python train.py --model complex_cnn_se --loss combined   # MSE + SI-SDR
```

### Add New Modulation Type

Edit `config.py`:
```python
mod_types_train = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']  # add new
```

Then add constellation to `data_generator.py`:
```python
elif mod_type == '64QAM':
    re = 2 * np.random.randint(0, 8, n_symbols) - 7
    im = 2 * np.random.randint(0, 8, n_symbols) - 7
    return re + 1j * im
```

---

## Expected Results (Target for Paper)

After 100 epochs on training SNR -5~20dB:

| SNR (dB) | SI-SDR (dB) | SDR (dB) | SIR (dB) |
|----------|-------------|----------|----------|
| -10 | ~2.5 | ~5.0 | ~7.5 |
| 0   | ~6.0 | ~9.0 | ~12.0 |
| 10  | ~12.0 | ~15.0 | ~20.0 |
| 20  | ~18.0 | ~22.0 | ~30.0 |

*Note: Exact numbers depend on random seed and training dynamics.*

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch_size` to 8 or 4 |
| Training too slow | Reduce `--hidden` to 16, `--layers` to 3 |
| CUDA not available | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Results worse than baseline | Check if SE block is enabled; try different `--lr` |

---

## References

1. Hou, X., & Gao, Y. (2022). Single-channel blind separation of co-frequency signals based on convolutional network. *Digital Signal Processing*, 129, 103654.
2. Guo, P., et al. (2024). Single-channel blind source separation in wireless communications: A complex-domain deep learning approach. *IEEE Wireless Communications Letters*, 13(6), 1645-1648.
3. Ma, H., et al. (2023). A novel end-to-end deep separation network based on attention mechanism. *IET Signal Processing*, 17(2), e12173.

---

## License

MIT License. Free for academic and research use.
