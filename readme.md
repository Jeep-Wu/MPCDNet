# MPCDNet — Deployment Guide

## 1. Environment Setup

Create a `.env` file:

```bash
export nnUNet_raw=<path to nnUNet_raw>
export nnUNet_preprocessed=<path to nnUNet_preprocessed>
export nnUNet_results=<path to nnUNet_results>
```

Load the variables:

```bash
source .env
```

---

## 2. Training

Use `starter.py` with `MPCDNetTrainer`:

```bash
python starter.py 563 2d -tr MPCDNetTrainer -c 0,1,2
```

* `563`: dataset ID
* `2d`: configuration (`2d`, `3d_fullres`, etc.)
* `-c 0,1,2`: train 3 models (indices 0, 1, 2)

---

## 3. Inference

Run `nnUNetv2_predict` with `MPCDNetTrainer` and `MPCDNetPredictor`:

```bash
nnUNetv2_predict -i imagesTs -o Output -d 563 -c 2d -f 0 \
-chk 'checkpoint_best.pth' -tr MPCDNetTrainer -pr MPCDNetPredictor \
--save_probabilities --save_raw
```

* `-chk`: checkpoint file to use
* `--save_probabilities`: save probability maps
* `--save_raw`: save raw outputs

---

## 4. Notes

* Ensure `.env` paths follow the nnU-Net directory structure.
* Use matching dataset ID and configuration for both training and inference.
* Confirm CUDA, PyTorch, and nnU-Net compatibility before running.
