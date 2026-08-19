# Fine-tuning the pre-trained RoNIN ResNet on the IMUNet dataset

`finetune_imunet.py` takes the RoNIN authors' pre-trained ResNet18 checkpoint
(`models/ronin_resnet_model.pt`) and adapts it to the IMUNet dataset in
`Datasets/IMUNet_dataset`.

The checkpoint transfers without any surgery: it was trained with `GlobSpeedSequence`
features (global-frame gyroscope + accelerometer, 200-sample window) regressing global 2D
velocity, and `ProposedSequence` — the reader for the IMUNet dataset — produces exactly the
same layout. All 132 tensors load with no missing or unexpected keys.

## Environment

PyTorch's CUDA wheels fail to load on this machine under Python 3.14 (`WinError 1114` on
`c10.dll`, also with a clean venv and on 3.12), and torch 2.11/2.13 CUDA builds fail the same
way. The combination that works is **Python 3.12 + torch 2.7.1+cu126**:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe -m pip install "numpy<2.3" "pandas<3" scipy matplotlib numpy-quaternion tensorboard tqdm h5py
```

Two upstream files needed a small fix to import on a current scientific stack: `utils.py`
imported `numba.jit` and `scipy.ndimage.filters.gaussian_filter1d` without ever using them
(the latter no longer exists in SciPy ≥ 1.15), and `model_resnet1d.py` imported `pthflops` at
module level although it is only used in its `__main__` demo.

## The global-frame problem in the ARCore half of the dataset

`ProposedSequence` fixes the entire feature frame from a single sample, `ori_*[0]`:

```python
init_rotor = init_tango_ori * game_rv[0].conj()
ori        = init_rotor * game_rv
```

For the Lenovo Tango sequences this is correct. For the sequences recorded with the
Android/ARCore pipeline it is not: `Datasets/proposed/read_data_s10.py` remaps the recorded
position to `(x, -z, y)` but leaves the pose quaternion in ARCore's own frame, so `init_rotor`
becomes a wrong constant rotation. Measured over all 126 sequences, the angle between gravity
in the feature frame and +Z is:

| group | sequences | gravity tilt |
|---|---|---|
| Tango | 49 | 0.1°–1.8° |
| ARCore (S10 / S21 / Xiaomi) | 77 | 34.9°–115.6° |

Features and targets of the ARCore sequences therefore live in two frames separated by an
arbitrary per-sequence rotation — including an arbitrary yaw. Yaw is not learnable here:
velocity regression from a global-frame window is yaw-equivariant, which is precisely the
symmetry `RandomHoriRotate` exploits for augmentation. Left uncorrected it is irreducible
noise, and it also makes that augmentation invalid for those sequences.

`frame_alignment.py` fixes this. Detection is ground-truth free (gravity more than
`--align_tilt_deg`, default 20°, away from +Z — this separates the two groups cleanly). The
correction is a calibration step that uses the sequence's ground-truth positions: specific
force in the global frame must satisfy

```
R @ f_feature  ==  a_gt + (0, 0, 9.81)
```

so `R` follows in closed form from Kabsch/SVD — gravity pins the two tilt degrees of freedom,
the walking accelerations pin yaw. The estimated rotations are cached in
`Datasets/IMUNet_dataset/frame_alignment.json`. Pass `--no_align_frame` to reproduce the
uncorrected pipeline.

The effect on the pre-trained model, with no training at all, on the 36 test sequences:

| | avg loss | avg ATE | avg RTE |
|---|---|---|---|
| RoNIN zero-shot, uncorrected | 0.4298 | 26.75 m | 21.37 m |
| RoNIN zero-shot, frame-corrected | 0.0887 | 9.18 m | 7.91 m |

## Splits

`Datasets/IMUNet_dataset/list_train.txt` (90 sequences) and `list_test.txt` (36) ship with the
dataset and together cover all 126 sequences on disk. The script holds out a validation set
from the training list (`--val_ratio`, default 0.12), drawing whole sequences and spreading
them over every subject/device group so each combination is represented. The split is
deterministic given `--seed` and is written to `config.json` in the output directory.

## Usage

```bash
cd RONIN_torch

# fine-tune (default: all layers, lr 1e-4, 50 epochs, early stop after 12 stale epochs)
../.venv/Scripts/python.exe finetune_imunet.py --mode train --save_plots

# evaluate any checkpoint on the test list
../.venv/Scripts/python.exe finetune_imunet.py --mode eval \
    --model_path Train_out/ResNet/imunet_finetune/checkpoints/checkpoint_best.pt \
    --out_dir Test_out/imunet/finetuned
```

Useful flags: `--freeze_stages N` (0 = fine-tune everything, 1 = freeze the stem, 2..5 = stem
plus the first 1..4 residual groups), `--pretrained ""` to train from scratch for comparison,
`--lr`, `--batch_size`, `--epochs`, `--early_stop`, `--cpu`.

## Results

All numbers are on the 36 sequences of `list_test.txt`, averaged per sequence; ATE/RTE in
metres. Both fine-tuning runs start from the frame-corrected data and fine-tune every layer on
an RTX 2080 Super (~68 s/epoch, 368 526 training windows).

| run | avg loss | avg ATE | avg RTE |
|---|---|---|---|
| RoNIN zero-shot, uncorrected frames | 0.4298 | 26.75 | 21.37 |
| RoNIN zero-shot, frame-corrected | 0.0887 | 9.18 | 7.91 |
| fine-tuned, lr 1e-4 | 0.0416 | 6.35 | 4.57 |
| **fine-tuned, lr 3e-5** | **0.0415** | **5.88** | **4.45** |

Both runs reach their best validation loss almost immediately — epoch 1 at lr 1e-4 (0.0363)
and epoch 2 at lr 3e-5 (0.0359) — and then overfit while training loss keeps falling. The
pre-trained RoNIN features are already close to optimal for this data once the frames agree,
so there is little left for the fine-tuning to do; lowering the learning rate further or
freezing stages is unlikely to move the number much. What dominates the remaining error is a
handful of sequences with poor ground truth, chiefly `Outdoor_Subject_2_Xiaomi_2` (ATE 30.0 m)
and `Outdoor_Subject_1_Xiaomi_10` (ATE 20.5 m); the median test sequence is at roughly 4 m
ATE.

Checkpoints: `Train_out/ResNet/imunet_finetune/checkpoints/checkpoint_best.pt` (lr 1e-4) and
`Train_out/ResNet/imunet_finetune_lr3e-5/checkpoints/checkpoint_best.pt` (lr 3e-5, the better
of the two). Training logs are in `finetune_log.txt` and `finetune_log_lr3e-5.txt` at the
repository root.

## Training on your own dataset

### Files you need

Per sequence, one directory under `--root_dir` named after the sequence, containing
`processed/data.csv` (or `processed/data.pkl`) with these columns — `ProposedSequence` reads
nothing else:

| column | meaning |
|---|---|
| `time` | timestamp in **nanoseconds** (divided by 1e9 on load) |
| `gyro_x/y/z` | raw angular rate, device frame |
| `acce_x/y/z` | raw acceleration **including gravity**, device frame |
| `pos_x/y/z` | ground-truth position; x-y must be the **horizontal** plane |
| `ori_w/x/y/z` | ground-truth attitude, **same frame as `pos_*`** |
| `rv_w/x/y/z` | Android game rotation vector |

Plus `list_train.txt` and `list_test.txt` at the root of `--root_dir`, one sequence name per
line. Sampling is assumed to be 200 Hz.

The two columns that matter most are `pos_*` and `ori_*`, because `ProposedSequence` builds the
feature frame from a single sample of them:

```python
init_rotor = ori_[0] * rv_[0].conj()   # constant, calibrated on ONE sample
ori        = init_rotor * rv_
```

This only yields a learnable problem under two assumptions: `ori_*` and `pos_*` live in the
same frame, and that frame is Z-up (the target is `pos_[:, :2]`). IMUNet's own ARCore half
breaks the first one. A new dataset can break either, silently.

### Step by step

**1. Check the frames before anything else.**

```bash
cd RONIN_torch
../.venv/Scripts/python.exe check_dataset_frames.py --root_dir <seu_dataset> --json report.json
```

`check_dataset_frames.py` runs four per-sequence checks (gravity tilt, which `pos_` axis is
vertical, the `init_rotor` residual between `ori_` and `rv_`, and the residual rotation from
`frame_alignment.estimate_alignment` split into tilt and yaw) and prints a verdict per sequence
plus a recommendation. Run against the IMUNet dataset it reproduces the known split exactly —
the same 77 broken and 49 correct sequences as `frame_alignment.json`, with no access to it.

**2. Act on the verdict.**

| verdict | meaning | what to do |
|---|---|---|
| `OK` | features and targets share a Z-up frame | nothing; train with `--no_align_frame` |
| `FRAME-INCONSISTENTE` | `ori_` and `pos_` in different frames | fix at the source, or use `frame_alignment.py` |
| `NAO-Z-UP` | ground-truth frame is not Z-up | rotate `pos_` **and** `ori_` together to Z-up |
| `ORI-RV-INCOMPATIVEL` | `ori_` and `rv_` differ by a non-constant rotation | `init_rotor` cannot work; check the recording |
| `YAW-SUSPEITO` | yaw error above the estimator noise floor | investigate; the tilt gate will not catch it |
| `DERIVA` | `rv_` and `ori_` drift apart over the sequence | consider a least-squares `init_rotor` |
| `DADOS` | NaN/inf, or non-monotonic timestamps | fix first; other checks are unreliable |

Fixing `FRAME-INCONSISTENTE` **at the source is strictly better** than using
`frame_alignment.py`: apply the same axis remap to `pos_` and `ori_`, or to neither. That is
exact, needs no ground truth, and avoids the yaw noise below. Reach for `frame_alignment.py`
only when you cannot regenerate the data.

**3. If you use `frame_alignment.py`, set the gate deliberately.** Its detection is a gravity
tilt threshold (`--align_tilt_deg`, default 20°), which works on IMUNet because the two groups
are cleanly separated (0.09-1.79° vs 34.86-115.63°). `check_dataset_frames.py` prints the
usable range for your data. Two limits are worth knowing:

- **The tilt gate is not an optimisation, it is a guard.** Measured on IMUNet sequences that
  are already correct, `estimate_alignment` returns 3-18° of *spurious yaw* (its tilt stays
  accurate to <2°). Applying the correction unconditionally would inject that into good data.
- Consequently the correction only fixes **gross** frame errors. A pure yaw offset of a few
  degrees is invisible to the tilt check and beyond the estimator's resolution.

**4. Check that the validation split is not empty.** `split_train_val` groups sequences by
`name.rsplit('_', 1)[0]` and keeps at least one member of each group in training. Names that do
not follow the `<Environment>_<Subject>_<Device>_<index>` convention — `seq001`, `walk01` —
put every sequence in a group of its own, which used to empty the validation set entirely. That
is far from harmless: with no validation loader the training loop never writes
`checkpoint_best.pt`, never steps `ReduceLROnPlateau` and never early-stops, so it runs all
`--epochs` at a fixed LR and leaves you only the last epoch. There is now an ungrouped random
fallback for that case, and an explicit warning if the validation set still ends up empty, but
the grouped split is the one you want — it guarantees every subject/device combination is
represented. Either name your sequences accordingly or pass an explicit `--val_list`, which
bypasses the grouping altogether.

**5. Train from scratch.** `--pretrained ""` skips `load_pretrained`:

```bash
../.venv/Scripts/python.exe finetune_imunet.py --mode train \
    --pretrained "" \
    --root_dir <seu_dataset> \
    --epochs 100 --lr 1e-4 --early_stop 20 \
    --out_dir Train_out/ResNet/meu_dataset --save_plots
```

Add `--no_align_frame` if step 1 reported everything `OK`, and `--val_list` if step 4 applies.

The 50-epoch / early-stop-12 defaults were tuned for fine-tuning, which converged at epoch 1-2;
from scratch needs considerably more. Watch `history.json`: training loss falling while
validation loss flattens means the dataset is too small for the 4.6 M-parameter network, and
the levers are `--freeze_stages` (not useful from scratch), a smaller `--lr`, or more data.
`RandomHoriRotate` is applied to the training split only and is the main regulariser — it
rotates each window and its target by a random yaw, which is valid precisely because global-frame
velocity regression is yaw-equivariant. That validity depends on step 1 passing; on sequences
whose frames disagree the augmentation is not just useless but actively wrong.

**6. Export.** `export_pte.py --checkpoint <seu_checkpoint>` (see below).

### A cheap improvement to `init_rotor`

Calibrating on one sample anchors the error at zero at t=0 and lets it grow. Measured on
correct IMUNet sequences, the residual between `init_rotor * rv_` and `ori_` goes from 0° at
t=0 to 2-6° by the end — real relative drift between the game rotation vector and the tracker.
Estimating `init_rotor` by least squares over the whole sequence (an average of rotations via
SVD, instead of `rv_[0]`) spreads that error instead of letting it accumulate. It costs
nothing and matters most on long sequences.

## Export to ExecuTorch (`.pte`)

`export_pte.py` turns a fine-tuned checkpoint into an ExecuTorch program for on-device
inference. The checkpoint is loaded with `strict=True`, the network is exported in `eval` mode
(BatchNorm in inference mode, the two Dropout layers become identities) and lowered to the
XNNPACK CPU delegate; `--no_xnnpack` emits a portable-kernel program instead.

```bash
cd RONIN_torch
../.venv-et/Scripts/python.exe export_pte.py
```

Default input is `--checkpoint Train_out/ResNet/imunet_finetune_lr3e-5/checkpoints/checkpoint_best.pt`
(the better of the two runs) and the output lands next to the run directory as `model.pte`
(17.7 MiB, 4 634 882 parameters). The signature is a single tensor `(1, 6, 200)` — global-frame
gyroscope then accelerometer, 200 samples — returning `(1, 2)`, the global 2D velocity in m/s.
The batch size is baked in; `--batch_size` changes it.

After writing the file the script loads it back through the ExecuTorch runtime and compares
against the eager model (`--no_verify` skips this). Agreement is ~1e-6; on eight real windows
from `Indoor_Subject_1_S10_1` the largest deviation was 6.7e-7.

### Export environment

ExecuTorch pins its own torch version, so it gets a **separate** virtualenv — installing it
into `.venv` would replace the torch 2.7.1+cu126 used for training:

```bash
py -3.12 -m venv .venv-et
.venv-et/Scripts/python.exe -m pip install executorch     # pulls torch 2.13.0+cpu
```

Two Windows-specific problems have to be worked around:

**`OSError: [WinError 1114]` on `c10.dll`.** This machine has no Visual C++ 2015-2022
redistributable installed — the only `msvcp140.dll` / `vcruntime140.dll` in `System32` are the
14.0.23506 copies left by the .NET runtime, and `vcruntime140_1.dll` is missing entirely.
torch ≥ 2.8 is built against 14.4x and its `c10.dll` imports all three, so initialisation
fails. (This, not the CUDA build, is the real cause of the failures noted under *Environment*
above; torch 2.7.1 works only because it was built with an older MSVC.) The fix used here is
`pip install msvc-runtime` plus a `sitecustomize.py` in the venv's `site-packages` that loads
the 14.44 DLLs by absolute path before anything else, so later resolutions by name bind to
them rather than to the System32 copies. Installing the redistributable system-wide
(`winget install Microsoft.VCRedist.2015+.x64`) fixes it properly and makes both the
`sitecustomize.py` and the `msvc-runtime` package unnecessary.

**`FileNotFoundError: [WinError 2]` during XNNPACK serialization.** executorch 1.4.1 looks for
a packaged resource named literally `flatc`, but on Windows the binary is `flatc.exe`; the
lookup misses and falls back to a bare `flatc` on `PATH`, which is only present when the venv
has been activated. `export_pte.py` sets `FLATC_EXECUTABLE` to the absolute path before
importing executorch, so it works however python is invoked.

Outputs land in `--out_dir` (default `Train_out/ResNet/imunet_finetune`):
`checkpoints/checkpoint_best.pt` and `checkpoint_latest.pt`, `config.json` with the resolved
splits, `history.json` with the per-epoch losses, `metrics_test.csv` with per-sequence
MSE/ATE/RTE, and — with `--save_plots` — one predicted-vs-ground-truth trajectory plot per
test sequence.
