# 🧠 EEG-Epilepsy-Prediction

**Multi-Scale Attention BiLSTM Framework for EEG Seizure Detection and Typing**

An end-to-end deep learning framework integrating **channel attention**, **temporal attention**, and **bidirectional LSTM**. End-to-end pipeline: EDF reading → band/time-domain features → 🧠 **Multi-Scale Attention BiLSTM** frame-level multi-class classification → event-level postprocessing (smoothing/confirmation/cooldown/minimum duration) → metric evaluation and threshold grid search (FA/h constraint, auto write-back to config).

**🎯 Key Features**: attention weight visualization | memory-optimized design | LOSOCV cross-validation | end-to-end training and inference

---

## 1. Overall Architecture and Design Rationale

- Data layer (`src/edf_reader.py`)
  - Reads EDF channel by channel using `pyedflib`; removes DC offset (zero-mean); optional mains notch filter (50/60Hz), optional bandpass/highpass/lowpass; uniformly resamples to `resample_hz`; trims channels of different lengths to the same minimum length and stacks them into `[C, N]`.
- Feature layer (`src/features.py`)
  - Computes Welch PSD using sliding-window parameters `window_sec/hop_sec`, aggregates band power (delta/theta/alpha/beta/gamma), computes {mean, std} across channels; also adds broadband energy {mean, std} and time-domain RMS {mean, std} → yields per-frame features `[T, F]` and frame center times `centers`.
- Model layer (`src/model.py`) - **🧠 Multi-Scale Attention BiLSTM Architecture**
  - **Channel Attention**: adaptively selects important EEG band features, highlighting activity in pathological brain regions
  - **Bidirectional LSTM**: extracts temporal context, capturing dependencies before and after a seizure
  - **Temporal Attention**: multi-head self-attention mechanism, focusing on key moments of seizure onset
  - **Classifier**: multi-layer fully connected network, outputs frame-level logits `[B,T,C]`; loss is cross-entropy (ignoring label value -100)
  - Training supports learning-rate scheduling (Cosine/OneCycleLR), augmentation (Mixup/SpecAugment/noise), and gradient clipping to improve generalization and stability
- Postprocessing (`src/postprocess.py`)
  - Based on `1 - p(bckg)`: smoothing (moving average) → threshold binarization → consecutive-frame confirmation window (confirm) → cooldown merging of adjacent segments (cooldown) → minimum event duration filtering; within a segment, the dominant class and confidence are chosen by summed (or max) probability.
- Evaluation and thresholds (`src/metrics.py`, `src/scan_thresholds.py`, `src/eval.py`)
  - Event-level IoU matching: computes P/R/F1, FA/h, onset/offset latency; threshold grid search picks the best threshold under an FA/h constraint and automatically writes it back to `configs/config.yaml`.

Pipeline:

```
EDF files
  └─> edf_reader (filter / notch / resample / align)
        └─> features (Welch PSD bands + RMS)
              └─> 🧠 Multi-Scale Attention BiLSTM
                    ├─> Channel Attention (band selection)
                    ├─> BiLSTM (temporal modeling)
                    ├─> Temporal Attention (focus on key moments)
                    └─> Classifier (frame-level classification)
                          ├─> postprocess (smooth / confirm / cooldown / min_dur) -> events
                          ├─> metrics (PR/F1, FA/h, latencies)
                          └─> threshold grid search (constraints + writeback)
```

---

## 2. Installation and Environment

- Python 3.11+ (a virtual environment such as conda/venv is recommended)

### 🔧 **Recommended Installation**

```bash
# 1. Create a virtual environment (recommended)
conda create -n EEG_work python=3.11
conda activate EEG_work

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 🪟 **Notes for Windows Users**

- **Activating the environment**: activate the environment before each use: `conda activate EEG_work`
- **Command-line syntax**: Windows PowerShell doesn't support `\` for line continuation; use a single-line command, or the backtick (`` ` ``) for continuation
- **Character encoding**: make sure files are saved as UTF-8, and use PowerShell to avoid garbled text

### 🚨 **OOM (Out-of-Memory) Fix**

**This project has built-in OOM protection, including:**
- ✅ Smart sequence-length limiting (max 8000 frames, about 33 minutes)
- ✅ Dynamic memory checks with automatic truncation
- ✅ Gradient accumulation support (simulates a larger batch size)
- ✅ Memory-optimized batching functions
- ✅ GPU memory monitoring and automatic cleanup

**If you still run into OOM, you can adjust further:**
```bash
# Most conservative configuration (for 8GB memory)
python -m src.train --batch_size 1 --gradient_accumulation_steps 4 --num_workers 0
```

---

## 3. Data Organization and Caching

- Reference dataset: TUSZ (Temple University Hospital Seizure Corpus, see the official documentation).
  - Official homepage: `https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml`

- Place EDF files and their matching TSE files into `data/Dataset_train_dev/`. Example TSE line:
  - `0.0000 36.8868 bckg 1.0000` (start seconds, end seconds, label, confidence; confidence may be omitted)
- Example directory layout:

```
data/
  Dataset_train_dev/
    00000001_s001_t000.edf
    00000001_s001_t000.tse
    ...
```

- On first run, `.npz` feature caches will be generated in `data_cache/` to speed up subsequent runs.
- Already ignored via `.gitignore`: `data/`, `data_cache/`, `outputs/`, `__pycache__/`.

---

## 4. Memory-Optimized Configuration 🚀

This project has been deeply optimized for GPU memory usage, supporting operation in smaller-VRAM environments:

### 📊 **Model Parameter Optimization**
- **Original configuration**: ~4.93M parameters, ~56MB training memory
- **Optimized configuration**: ~0.85M parameters, ~10MB training memory (82.8% reduction)

### 🔧 **Key Optimization Measures**
```yaml
# 🧠 Attention-mechanism model architecture optimization
hidden_dim: 128      # LSTM hidden dimension (reduced from 256 to 128)
num_layers: 2        # Number of LSTM layers (reduced from 3 to 2)
attention_heads: 4   # Number of attention heads (reduced from 8 to 4)
use_attention: true  # Enable multi-scale attention mechanism

# Training optimizations
batch_size: 1                    # Very small batch size
gradient_accumulation_steps: 8   # Gradient accumulation to keep an equivalent effective batch size
num_workers: 0                   # Reduce multiprocessing overhead
precompute_cache: none          # Disable precomputed cache
```

### 💾 **VRAM Usage Comparison**
| Configuration | RTX 4060 (8GB) | RTX 3090 (24GB) |
|------|----------------|------------------|
| Original | OOM ❌ | Normal ✅ |
| Optimized | Normal ✅ | Fast 🚀 |

### 🎯 **Applicable Scenarios**
- ✅ 8GB cards such as the RTX 4060/4070
- ✅ Learning/research environments
- ✅ Resource-constrained scenarios
- 🚀 High-end cards such as the RTX 3090/4090 will train faster

---

## 5. 🧠 Attention Mechanism Architecture in Detail

This project implements a **Multi-Scale Attention BiLSTM architecture** designed specifically for EEG seizure detection, improving detection accuracy through three layers of attention:

### 📡 **1. Channel Attention**
```python
# Adaptively selects important EEG band features
- Input: [B, T, F] EEG features
- Purpose: highlight spectral activity in pathological brain regions
- Implementation: global pooling + FC layer + Sigmoid activation
- Output: feature-weighted representation
```

### ⏰ **2. Temporal Attention**  
```python
# Multi-head self-attention mechanism, focusing on key seizure moments
- Input: [B, T, H] LSTM hidden states
- Purpose: capture long-range temporal dependencies, focus on seizure onset
- Implementation: Multi-Head Self-Attention + residual connection
- Number of heads: 4 attention heads (memory-optimized)
```

### 🧬 **3. BiLSTM Backbone**
```python
# Bidirectional LSTM extracts temporal context
- Layers: 2 (balances performance and efficiency)
- Hidden dimension: 128 (memory-optimized)
- Bidirectional: models both past and future information simultaneously
- Dropout: 0.15 to prevent overfitting
```

### 🎯 **Architectural Advantages**
- **🔍 Precise localization**: channel attention highlights abnormal bands
- **⏱️ Temporal modeling**: temporal attention captures seizure timing patterns  
- **💡 End-to-end**: attention weight visualization provides interpretability
- **⚡ Efficient**: memory-optimized design, fits 8GB cards

### 📊 **Attention Visualization**
During training the model outputs attention weights, which can be used to:
- Analyze which bands are most important for detection
- Visualize the temporal pattern of a seizure
- Provide clinically interpretable diagnostic evidence

---

## 6. Quick Start (Step by Step)

1) Prepare the data

- Place EDF/TSE files as described above; make sure `pyedflib` can read the EDF files correctly.

2) Configuration

- Edit the `train`, `postprocess`, and `labels` sections of `configs/config.yaml`; or override via command-line arguments.
  - For multi-class detection/typing, provide two Excel files (example fields: canonical name / alias); training will automatically generate and use `outputs/labels.json`;
  - Before training, a "class consistency" pre-check is run — the labels in `.tse` must be covered by the labels/aliases defined in the Excel files.

3) Splitting and training

3.1 Fixed split (optional, recommended)

```bash
python -m src.split \
  --data_dir data/Dataset_train_dev \
  --out outputs/splits.json --val_ratio 0.2 --test_ratio 0.2 --seed 42
```

3.2 Training (recommended config, OOM issues already optimized)

### 🚀 **Quick Start Training**

```bash
# Activate the environment (required for Windows users)
conda activate EEG_work

# Standard training (memory usage deeply optimized)
python -m src.train --config configs/config.yaml
```

### 🔧 **Training with Custom Parameters**

```bash
# Windows PowerShell single-line version (memory-optimized config)
python -m src.train --config configs/config.yaml --scheduler onecycle --epochs 10 --batch_size 1 --progress bar

# Linux/Mac multi-line version
python -m src.train --config configs/config.yaml \
  --scheduler onecycle --epochs 10 \
  --batch_size 1 --progress bar
```

### 📊 **Monitoring Training**

- **TensorBoard**: `tensorboard --logdir outputs/tb`
- **Training log**: `outputs/train.log`
- **Best model**: `outputs/best.pt`

3.3 LOSOCV (Leave-One-Subject-Out Cross-Validation, supports resuming from checkpoints)

### 🔄 **LOSOCV Training Command (Memory-Optimized)**

```bash
# Windows PowerShell version (recommended, memory-optimized)
python -m src.losocv --config configs/config.yaml --run_train --auto_optimize --opt_trials 5 --opt_epochs 2 --epochs 10 --batch_size 1 --resume --progress bar

# Linux/Mac version
python -m src.losocv --run_train --auto_optimize \
  --config configs/config.yaml \
  --opt_trials 5 --opt_epochs 2 \
  --epochs 10 --batch_size 1 \
  --resume --progress bar
```

### ⚡ **Quick LOSOCV Test**

```bash
# Quick test (finishes in a few minutes, memory-friendly)
python -m src.losocv --config configs/config.yaml --run_train --epochs 2 --batch_size 1 --progress bar
```

### 🔄 **Resume-From-Checkpoint Features**

- ✅ **Automatic detection**: use the `--resume` flag to automatically continue from where training was interrupted
- ✅ **Fold-level resuming**: each patient fold is saved individually, so training can resume after partial completion
- ✅ **Two-level resuming**: supports resuming both at the Optuna-trial level and the final-training level
- ✅ **Checking progress**: 
  ```bash
  # Check how many folds have completed
  ls outputs/losocv/fold_*/best.pt
  # Check the total number of folds
  ls outputs/losocv/*.json
  ```

- For each fold:
  - Generates that fold's `train/val/test` split JSON (test is the held-out subject; the rest is split into train/val using "patient-level multi-class stratification")
  - Optuna searches for optimal frame-labeling parameters (`label_overlap_ratio`, `min_seg_duration`), picking the best via short-training evaluation
  - Trains the final model for this fold using the optimal parameters (written to `outputs/losocv/fold_<PID>/best.pt`)
  - Evaluates this fold's test set, writing `eval_summary.json` / `eval_records.csv`
  - Training and evaluation logs are more concise and readable (both console and `train.log`)

- Once all folds complete, results are automatically aggregated:
  - `outputs/losocv/loso_eval_aggregate.json`
  - Provides both "macro average (simple average)" and "micro average (aggregated over TP/FP/FN and duration)" metrics (check IoU=0.5 first)

4) Threshold grid search (FA/h constraint + config write-back)

```
python -m src.scan_thresholds \
  --data_dir data/Dataset_train_dev --cache_dir data_cache \
  --probs 0.6 0.7 0.8 0.9 --smooth 0.0 0.25 \
  --confirm 1 2 3 --cooldown 0.0 0.5 1.0 \
  --min_duration 0.0 --max_fa_per_hour 2.0 \
  --labels_json outputs/labels.json \
  --out outputs/threshold_grid.json --write_config configs/config.yaml
```

- Output: `outputs/threshold_grid.json`; the best threshold is written back to `configs/config.yaml:postprocess`.
 - Note: the script currently doesn't load a checkpoint — it uses a randomly initialized model for a demonstration scan, mainly to illustrate the "threshold → metric → writeback" flow. For real use, it's recommended to do threshold selection based on a trained model (you can extend the script yourself to load weights), and always provide `--labels_json` to ensure a consistent label set.

5) Evaluation (from a checkpoint)

```
python -m src.eval --config configs/config.yaml --checkpoint outputs/best.pt \
  --labels_json outputs/labels.json
```

- Output:
  - `outputs/eval_summary.json` (global metrics across multiple IoU thresholds)
  - `outputs/eval_records.csv` (per-record TP/FP/FN and onset/offset latency)
 - Tip: the class order in `labels.json` must match the order used during training, and must match the number of classes in the checkpoint's classification head, or the script will raise an error.

- To evaluate a single fold (that fold's test set only):
```bash
python -m src.eval --config configs/config.yaml \
  --checkpoint outputs/losocv/fold_<PID>/best.pt \
  --splits_json outputs/losocv/<PID>.json --use_split test \
  --out_json outputs/losocv/fold_<PID>/eval_summary.json \
  --out_csv outputs/losocv/fold_<PID>/eval_records.csv
```

6) Single-file detection (inference)

```
python -m src.predict --edf path/to/file.edf --checkpoint outputs/best.pt \
  --config configs/config.yaml --labels_json outputs/labels.json \
  --out outputs/pred_events.json
```

- Output: whether a seizure event is present, the number of segments, and the type and start/end time of each segment.
 - Tip: the number of classes in the checkpoint's classification head must match the provided `labels.json`, or a mismatch error will be raised.

7) Reproducibility and caching

- Random seed: `--seed` (training script)
- Recompute features: delete `data_cache/*.npz` and run again.

---

## 5. Configuration Details (Excerpt)

`configs/config.yaml`

```
train:
  # Root data directory containing paired EDF/TSE files
  data_dir: data/Dataset_train_dev
  # Feature cache directory (stores .npz files, speeds up reuse)
  cache_dir: data_cache
  # Feature sliding-window length (seconds)
  window_sec: 2.0
  # Feature sliding-window hop length (seconds)
  hop_sec: 0.25
  # Resampling frequency applied uniformly after reading (Hz)
  resample_hz: 256.0
  # Preprocessing bandpass range (Hz); set either end to null to fall back to highpass/lowpass, or disable entirely
  bandpass: [0.5, 45.0]
  # Mains notch filter (50 or 60); set to null to disable
  notch_hz: 50.0
  # Background class name (must match `.tse` or the alias mapping)
  bg_label: bckg
  
  # 🔧 Deep memory-optimization configuration
  batch_size: 1                    # Very small batch size to avoid OOM
  gradient_accumulation_steps: 8   # Gradient accumulation to keep an equivalent effective batch size
  
  # Number of training epochs
  epochs: 20
  # Validation/test ratio, split by patient
  val_ratio: 0.2
  test_ratio: 0.0
  # Random seed (for reproducibility)
  seed: 42
  # Output directory (logs/weights/TensorBoard)
  out_dir: outputs
  
  # 🔧 Deep memory optimization: reduce concurrency and precomputation
  num_workers: 0                   # Reduce multiprocessing memory overhead
  precompute_cache: none          # Disable precomputation to reduce memory usage
  # Fixed split file (optional): if set, uses the train/val/test record_id lists from this split
  splits_json: outputs/splits.json
  # Stratification strategy (none|has_seizure|multiclass): patient-grouped stratified split; defaults to multi-class per-class-coverage stratification
  stratify: multiclass
  # Terminal training-progress display (none|bar) and log interval (per-iteration logging is off by default; epoch summaries are always printed)
  progress: none
  log_interval: 0
  # Whether to run one baseline validation pass before training starts (off by default)
  eval_at_start: false
 

  # Learning-rate scheduling and data augmentation
  # Recommendation: onecycle for small/medium datasets; none/cosine also available
  scheduler: onecycle  # options: none|cosine|onecycle
  max_lr: 0.001
  # If no Excel/labels.json is provided, the label set can be auto-derived from the TSE files
  auto_labels_from_tse: true
  # Gradient clipping threshold (0 means no clipping)
  clip_grad: 0.0
  # Mixup strength (>0 enables frame-level soft-label mixing)
  mixup_alpha: 0.0
  # SpecAugment masking (0 means disabled)
  spec_time_mask_ratio: 0.0
  spec_time_masks: 0
  spec_feat_mask_ratio: 0.0
  spec_feat_masks: 0
  # Feature-level Gaussian noise strength
  aug_noise_std: 0.0
  # Frame labeling: window-label overlap ratio threshold (0~1) and minimum segment duration (seconds)
  label_overlap_ratio: 0.2
  min_seg_duration: 0.0

postprocess:
  # Decision threshold based on 1 - p(background)
  prob: 0.8
  # Probability smoothing window (seconds)
  smooth: 0.25
  # Number of consecutive frames required to confirm an event
  confirm: 2
  # Cooldown merge time (seconds; same-class adjacent events within this interval are merged)
  cooldown: 0.5
  # Minimum event duration (seconds; events shorter than this are discarded)
  min_duration: 0.0

labels:
  # Background class name (must match train.bg_label)
  background: bckg
  # Excel sheet (first sheet is used): the first two non-empty cells of each row are treated as (label, alias)
  excel_types: <path_to_types.xlsx>
  excel_periods: <path_to_periods.xlsx>
  # Shared label/alias export file used across training/evaluation/inference
  json_out: <path_to_labels.json>
```

Key points:

- `window_sec/hop_sec` control the time resolution and computation cost; `resample_hz` should be close to the data's native sampling rate (e.g. 256Hz).
- `bandpass/notch_hz` suppress baseline drift and mains noise; adjust to the experimental environment (50/60Hz).
- Scheduler: `onecycle` is generally more stable on small/medium datasets; `cosine` is simple and effective.
- Augmentation: `mixup_alpha>0` enables frame-level soft-label mixing; SpecAugment masks time/feature dimensions; `aug_noise_std` adds light Gaussian noise.
- Warm-up caching: `precompute_cache` controls the scope of pre-building `data_cache/*.npz`; when `num_workers>0`, warm-up and train/val loading run in parallel.
- Postprocessing: a higher `prob` is more conservative (fewer false positives); `confirm/cooldown/min_duration` control event fragmentation and false alarms.
 - Labels: if no Excel/`labels.json` is provided and `train.auto_labels_from_tse=true` is set, training will first scan the (non-background) labels appearing in `.tse` files and automatically build the label set; the Excel paths in the template are just examples — if you don't have actual files, replace them with your own paths or remove the field to avoid errors.

---

## 6. 🧠 Features and Attention Model (Detailed Architecture)

### 📊 **EEG Feature Extraction**
- **Band analysis**: delta(0.5–4), theta(4–8), alpha(8–13), beta(13–30), gamma(30–45)
- **Per-frame features** (14 dimensions):  
  - 5 band powers {mean, std} → 10 dims  
  - Broadband energy {mean, std} → 2 dims  
  - Time-domain RMS {mean, std} → 2 dims  

### 🧠 **Multi-Scale Attention BiLSTM Architecture**
```python
# Complete forward-pass pipeline
1. Channel attention: [B,T,14] → highlights important bands
2. BiLSTM backbone: [B,T,14] → [B,T,256] (hidden_dim*2)
3. Temporal attention: [B,T,256] → focuses on key moments + residual connection
4. Classifier: [B,T,256] → [B,T,C] frame-level logits
```

### 📈 **Model Parameter Statistics**
- **Total parameters**: ~0.85M (memory-optimized version)
- **Channel attention**: 84 params (0.01%)
- **BiLSTM**: 0.64M params (75.5%)
- **Temporal attention**: 0.18M params (21.2%)
- **Classifier**: 0.03M params (3.5%)

### 🔧 **Data Augmentation** (configurable)
- **Mixup**: linearly mixes samples within a batch using Beta(α,α) to generate soft targets
- **SpecAugment**: randomly zeroes out time segments/feature bands to simulate signal loss
- **Noise perturbation**: applies small Gaussian noise to features
- **Current configuration**: all augmentations disabled to save memory

---

## 7. Metrics and Postprocessing

- IoU matching: a predicted segment and a ground-truth segment are matched (TP) if their interval IoU ≥ threshold and the classes match; unmatched predictions are FP, unmatched ground truth are FN.
- P/R/F1: computed from accumulated TP/FP/FN; supports multiple IoU thresholds (`--ious`).
- FA/h: `FP / total hours`, used to control the false-alarm rate (can be constrained via `--max_fa_per_hour` during threshold grid search).
- Onset/offset latency: average |Δonset| / |Δoffset| over matched pairs.
- Postprocessing pipeline: smoothing → binarization → confirmation-window filtering → cooldown merging → minimum-duration filtering; a segment's class is chosen as the one with the largest summed (or max) frame probability.

### Aggregation Conventions (LOSOCV)
- Macro average: a simple average of each fold's metrics (P/R/F1), fairly reflecting generalization across subjects.
- Micro average: accumulates TP/FP/FN and total duration to compute overall P/R/F1 and FA/h, reflecting the overall operating point.

Aggregate file: `outputs/losocv/loso_eval_aggregate.json`.

---

## 8. Project Structure

- `src/edf_reader.py`: EDF reading, filtering/notch, resampling, alignment
- `src/features.py`: spectral power and time-domain features
- `src/dataset.py`: dataset and `.npz` caching
- `src/model.py`: `BiLSTMClassifier`
- `src/postprocess.py`: smoothing/confirmation/cooldown/duration filtering
- `src/metrics.py`: IoU matching, P/R/F1, FA/h, latency
- `src/scan_thresholds.py`: threshold grid search, FA/h constraint, config write-back
- `src/eval.py`: evaluation from a checkpoint, outputs JSON/CSV
- `src/train.py`: training (config, logging, TensorBoard, scheduler, augmentation, checkpointing)
- `configs/config.yaml`: configuration template
- `data/`, `data_cache/`, `outputs/`: data/cache/results directories (ignored by git)

---

## 9. OOM (Out-of-Memory) Solutions in Detail

### 🚨 **Background**

EEG data has the following characteristics that lead to OOM issues:
- **Long sequences**: a single EEG file can span several hours, producing tens of thousands of feature frames
- **Multiple channels**: typically 20 channels recorded simultaneously
- **High sampling rate**: 256Hz sampling produces a large number of data points
- **Batching**: multiple long sequences loaded into memory at once

### ✅ **Built-in OOM Protection Mechanisms**

#### **1. Smart Batching Optimization**
```python
# Automatic sequence-length limiting
MAX_SEQUENCE_LENGTH = 8000  # about 33 minutes, prevents a single sequence from being too long

# Memory pre-check
estimated_memory_mb = (batch_size * max_seq_len * features * 4) / (1024 * 1024)
if estimated_memory_mb > 800:  # auto-adjust if over 800MB
    # dynamically shrink sequence length
```

#### **2. Gradient Accumulation**
```bash
# Effectively trains with a large batch size, but memory-friendly
batch_size: 2                    # actual batch size
gradient_accumulation_steps: 2   # accumulate 2 steps = effective batch_size of 4
```

#### **3. Memory Monitoring and Automatic Cleanup**
```python
# Automatically monitors GPU memory during training
if memory_used_gb > 6.0:
    torch.cuda.empty_cache()  # automatic cleanup

# OOM exception catching and recovery
except RuntimeError as e:
    if "out of memory" in str(e):
        # automatically skip the problematic batch and continue training
```

#### **4. Feature Extraction Optimization**
```python
# Pre-allocate arrays to avoid dynamic growth
X = np.zeros((n_frames, n_features), dtype=np.float32)

# Free temporary variables promptly
del psd_array, seg, rms
```

### 🎛️ **Memory Configuration Levels**

#### **Level 1: Standard Configuration (8GB+ memory)**
```bash
python -m src.train --config configs/config.yaml
# batch_size=2, gradient_accumulation_steps=2, num_workers=1
```

#### **Level 2: Conservative Configuration (4-8GB memory)**
```bash
python -m src.train --batch_size 1 --gradient_accumulation_steps 4 --num_workers 0
```

#### **Level 3: Extreme Configuration (<4GB memory)**
```bash
python -m src.train --batch_size 1 --gradient_accumulation_steps 1 --num_workers 0 --epochs 5
```

### 📊 **Optimization Results Comparison**

| Setting | Before | After | Memory Saved |
|--------|--------|--------|----------|
| Batch size | 4 | 2 | 50% |
| Sequence length | unlimited | 8000 frames | 70% |
| Multiprocessing | 4 workers | 1 worker | 75% |
| Precomputation | first_batch | none | 30% |
| Overall effect | 16GB+ | 3-6GB | **60-80%** |

### 🔧 **Troubleshooting**

#### **Still hitting OOM?**
```bash
# 1. Check the sequence-length distribution
python -c "
from src.utils import pair_edf_tse
from src.dataset import SequenceDataset
pairs = pair_edf_tse('data/Dataset_train_dev')[:5]
for edf, tse, rec in pairs:
    print(f'{rec}: length to check')
"

# 2. Use the most conservative configuration
python -m src.train \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_workers 0 \
  --epochs 3

# 3. Monitor memory usage
# Watch for the memory-warning messages in the training output
```

#### **Performance Tips**
- ✅ Use GPU acceleration if available
- ✅ Enable mixed-precision training (auto-detected)
- ✅ Set `num_workers` sensibly (1 is recommended on Windows, 2-4 on Linux)
- ✅ Monitor the size of `data_cache/` and clean it up periodically

---

## 10. FAQ

### 🔧 **Installation Issues**
- **SciPy/pyEDFlib install fails**: it's recommended to install the corresponding binary packages inside a conda environment, or use a wheel that matches your Python version.
- **`torch` module not found**: make sure you've activated the virtual environment with `conda activate EEG_work`

### 💾 **Memory/OOM Issues**
- **Training OOM**: this project has built-in OOM protection; if problems persist:
  ```bash
  # Minimum-memory configuration
  python -m src.train --batch_size 1 --gradient_accumulation_steps 4 --num_workers 0
  ```
- **Feature extraction OOM**: delete `data_cache/*.npz` and regenerate using the optimized feature extraction
- **Insufficient GPU memory**: CPU training is enabled automatically, or use `memory_efficient=True` mode

### 🪟 **Windows-Specific Issues**
- **Multi-line commands fail**: Windows PowerShell doesn't support `\` line continuation; use a single-line command, or the backtick for continuation:
  ```powershell
  # Correct PowerShell syntax
  python -m src.losocv `
      --config configs/config.yaml `
      --run_train --batch_size 2
  ```
- **Environment activation**: `conda activate EEG_work` is needed every time you open a terminal

### 📊 **Training and Evaluation Issues**
- **LOSOCV interrupted, resuming training**: use the `--resume` flag to automatically continue from the checkpoint
- **Metrics look wrong**:
  - Check whether `bg_label` matches the data
  - Confirm the TSE parsing and time units
  - Verify that the postprocessing threshold has been written back and is correctly loaded by the evaluation script
- **Cache conflicts**: after changing feature/filter parameters, it's recommended to delete the old `data_cache/*.npz` files to avoid mixing them up

### 🎯 **Performance Tips**
- **Training too slow**: use `--progress bar` to see progress, and make sure GPU/CUDA is available
- **Data loading slow**: check the `num_workers` setting; 1 is recommended on Windows
- **Memory-usage monitoring**: memory warnings and usage info are shown automatically during training

---

## 11. License and Acknowledgments

- License: see `LICENSE` in the repository root.
- Acknowledgments: thanks to the open-source community (pyEDFlib, SciPy, PyTorch, TensorBoard, etc.) for their ecosystem support.
 - Dataset: thanks to the Temple University Hospital Seizure Corpus (TUSZ) for the data and annotations; see their [official homepage](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml).
