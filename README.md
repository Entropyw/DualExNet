# NTIRE Dn50 Submission Package - Team 02 (DualExNet)

This folder is the final submission-ready package for NTIRE Dn50 using team ID `02`.

## Model
- Model name: `DualExNet`
- Team model file: `models/team02_DualExNet.py` (single-file model as required)
- Team checkpoint: `model_zoo/DualExNet.pth`

Architecture summary:
- Two-branch denoiser combining a local feature branch (window and shifted-window attention) and a global feature branch (MDTA + GDFN blocks).
- A learnable fusion gate adaptively combines the two branch outputs pixel-wise.
- Inference in this package enables x8 self-ensemble in model loading for stronger denoising robustness.

## Folder Structure (Submission)
- `models/team02_DualExNet.py`
- `model_zoo/DualExNet.pth`
- `test_demo.py` (contains `model_id == 2` loading path)
- `run.sh` (default runs model 02)
- `utils/`

## Dependencies
Recommended environment:
- Python 3.10+
- PyTorch 2.0+
- torchvision
- numpy
- opencv-python

Install dependencies:
```bash
pip install torch torchvision numpy opencv-python
```

## Run Inference
Edit `--data_dir` and `--save_dir` to your local challenge paths, then run:

```bash
bash run.sh
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0 python test_demo.py \
  --model_id 2 \
  --data_dir ./NTIRE2025_Challenge/input \
  --save_dir ./NTIRE2025_Challenge/results
```

## Notes for Organizers
- `model_id=2` is implemented in `test_demo.py`.
- Input data range is set to `1.0`.
- The checkpoint is loaded from `model_zoo/DualExNet.pth`.

## Clone Command
```bash
git clone https://github.com/Entropyw/DualExNet.git
```

If the repository name differs on GitHub, replace the URL accordingly before submission.
