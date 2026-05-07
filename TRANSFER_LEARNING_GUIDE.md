# ARCADE → SSVS Transfer Learning Workflow

## Overview

This guide describes how to use ARCADE pretraining to improve vessel segmentation on SSVS data.

### Two-Stage Pipeline

```
Stage 1: ARCADE Pretraining (Supervised)
├─ Input: ARCADE images + labeled vessel/stenosis masks (COCO format)
├─ Model: G_A generator with BCE loss
├─ Output: arcade_pretrained_G_A.pth weights
│
Stage 2: SSVS Finetuning (Cycle-Consistency)
├─ Input: SSVS images (unpaired A/B/C domains)
├─ Model: USSEGModel with G_A initialized from Stage 1
├─ Output: Final trained vessel segmentation model
└─ Result: Better vessel segmentation from domain adaptation
```

---

## Stage 1: Supervised Pretraining on ARCADE

### Step 1.1: Prepare ARCADE Data

Your ARCADE data should be structured as:
```
datasets/arcade/
├── train/
│   ├── images/     (1000 × 512×512 PNG files: 1.png, 2.png, ...)
│   └── annotations/
│       └── train.json  (COCO format with vessel + stenosis masks)
├── val/
│   ├── images/     (200 files)
│   └── annotations/train.json
└── test/
    ├── images/     (300 files)
    └── annotations/test.json
```

### Step 1.2: Create a Symlink to ARCADE Data

```bash
# Option A: Symbolic link (recommended)
cd C:\monai-projects\SSVS\datasets
mklink /D arcade "C:\monai-projects\vascular_proto\data_raw\ARCADE\syntax"

# Or manually copy
xcopy "C:\monai-projects\vascular_proto\data_raw\ARCADE\syntax" "C:\monai-projects\SSVS\datasets\arcade" /E
```

Verify:
```bash
ls ./datasets/arcade/train/images | head -5   # Should show 1.png, 2.png, ...
```

### Step 1.3: Train Stage 1 (Supervised on ARCADE)

```bash
# Activate environment
conda activate ssv_modern_py310

# Run pretraining
python train.py \
  --dataroot ./datasets/arcade \
  --name arcade_pretrain \
  --model arcade_supervision \
  --dataset_mode arcade \
  --arcade_mask_type vessel \
  --n_epochs 50 \
  --n_epochs_decay 0 \
  --batch_size 4 \
  --gpu_ids 0 \
  --lambda_vessel 1.0 \
  --lr 0.0002 \
  --input_nc 1 \
  --output_nc 1 \
  --display_freq 20 \
  --print_freq 5 \
  --save_latest_freq 25 \
  --display_env arcade_pretrain
```

**Expected Output:**
- Training log: `./checkpoints/arcade_pretrain/loss_log.txt`
- Best weights: `./checkpoints/arcade_pretrain/latest_net_G_A.pth`
- Visualization: Visdom + TensorBoard on default ports

**Stopping Criterion:**
- Watch validation loss (TensorBoard)
- Typically converges in 30-50 epochs for segmentation
- When validation metric plateaus, stop and move to Stage 2

### Step 1.4: Evaluate Stage 1 (Optional)

```bash
# Test on ARCADE test set
python -c "
import torch
from models.arcade_supervision_model import ARCADESupervisionModel
from options.train_options import TrainOptions

opt = TrainOptions().parse()
opt.name = 'arcade_pretrain'
opt.phase = 'test'
opt.isTrain = False

model = ARCADESupervisionModel(opt)
model.load_networks('latest')

# Compute metrics on test set...
"
```

---

## Stage 2: Cycle-Consistency Finetuning on SSVS

### Step 2.1: Initialize USSEGModel with ARCADE Weights

After Stage 1 completes, copy the pretrained weights:

```bash
# Copy ARCADE pretraining weights
cp ./checkpoints/arcade_pretrain/latest_net_G_A.pth \
   ./checkpoints/ssvs_finetuned/latest_net_G_A.pth
```

### Step 2.2: Finetune on SSVS with Transfer Learning

```bash
# Finetune with ARCADE-initialized weights
python train.py \
  --dataroot ./datasets/ssv \
  --name ssvs_finetuned \
  --model usseg \
  --dataset_mode usseg \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --batch_size 1 \
  --gpu_ids 0 \
  --lambda_A 10.0 \
  --lambda_B 10.0 \
  --lambda_identity 0 \
  --continue_train \
  --load_epoch latest \
  --display_freq 20 \
  --print_freq 5 \
  --save_latest_freq 25 \
  --display_env ssvs_finetuned \
  --lr 0.00005 \
  --lr_policy linear \
  --n_epochs_constant 50
```

**Key Differences from Cold-Start Training:**

| Parameter | Cold Start | Transfer Learning |
|-----------|-----------|-------------------|
| Initial weights | Random | ARCADE pretrained |
| Learning rate | 0.0002 | 0.00005 (lower) |
| Warmup epochs | N/A | First 20-30 epochs |
| Convergence speed | Slower | Faster (pre-learned features) |
| Final quality | Baseline | Improved (domain-adapted) |

### Step 2.3: Monitor Training

Open in separate terminals:

```bash
# Terminal 1: Watch training losses
Get-Content .\checkpoints\ssvs_finetuned\loss_log.txt -Tail 20 -Wait

# Terminal 2: Visdom dashboard
python -m visdom.server

# Terminal 3: TensorBoard
tensorboard --logdir ./checkpoints/ssvs_finetuned/tensorboard --port 6007
```

**Convergence Signs:**
- Cycle losses plateau around epoch 5-10
- G_A and G_B losses become stable
- Visual validation: Check HTML gallery for smooth vessel predictions

---

## Step 3: Checkpoint Selection & Inference

### Step 3.1: Identify Best Checkpoint

After training completes:

```bash
# Find best checkpoint by lowest cycle loss
python -c "
import re
with open('./checkpoints/ssvs_finetuned/loss_log.txt') as f:
    lines = f.readlines()
    best_loss = float('inf')
    best_epoch = 0
    for line in lines[-1000:]:  # Last 1000 lines
        match = re.search(r'cycle_A: (\d+\.\d+)', line)
        if match:
            loss = float(match.group(1))
            if loss < best_loss:
                best_loss = loss
                best_epoch = int(re.search(r'epoch:\s*(\d+)', line).group(1))
    print(f'Best epoch: {best_epoch}, Loss: {best_loss:.4f}')
"
```

### Step 3.2: Copy Best Checkpoint

```bash
# Copy best epoch weights
cp ./checkpoints/ssvs_finetuned/net_G_A_epoch_50.pth \
   ./checkpoints/ssvs_finetuned/best_net_G_A.pth
```

### Step 3.3: Inference on New Data

```python
import torch
from models.usseg_model import USSEGModel
from options.test_options import TestOptions
from PIL import Image
import numpy as np

# Load model
opt = TestOptions().parse()
opt.name = 'ssvs_finetuned'
opt.checkpoints_dir = './checkpoints'

model = USSEGModel(opt)
model.load_networks('best')
model.eval()

# Test on a single image
test_image = Image.open('path/to/test_image.png').convert('L')
test_image = np.array(test_image).astype(np.float32) / 255.0
test_image = torch.from_numpy(test_image).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
test_image = test_image * 2 - 1  # Normalize to [-1, 1]

with torch.no_grad():
    fake_B = model.netG_A(test_image)
    vessel_mask = torch.sigmoid(fake_B)

# Save result
result_np = vessel_mask[0, 0].cpu().numpy()
result_img = Image.fromarray((result_np * 255).astype(np.uint8))
result_img.save('vessel_prediction.png')
```

---

## Troubleshooting & Tips

### Issue 1: ARCADE Pretraining Loss Not Decreasing

**Symptoms:** Loss stays constant, doesn't drop
- Check image/mask alignment: `print(image.shape, vessel_mask.shape)`
- Verify masks are binary (0 and 1 only): `print(vessel_mask.min(), vessel_mask.max())`
- Reduce learning rate further: `--lr 0.0001`

### Issue 2: Finetuning Diverges (Losses Explode)

**Symptoms:** Cycle losses shoot to 1e6+ after first epoch
- Use **very small learning rate**: `--lr 0.00001`
- Reduce `lambda_A` and `lambda_B` initially: `--lambda_A 1.0 --lambda_B 1.0`
- Gradually increase cycle-consistency weights over epochs (requires code modification)

### Issue 3: Poor Generalization After Transfer Learning

**Possible Causes:**
- ARCADE domain too different from SSVS
- Overfitting to SSVS after pretraining (underfitting cycle-consistency)
- Too few SSVS training samples

**Fixes:**
- Use **shorter finetuning** (fewer epochs)
- Increase `--lambda_A` / `--lambda_B` to emphasize cycle-consistency
- Add augmentation: `--no_flip False --normalize_image`

### Issue 4: Memory Issues During ARCADE Training

**Solution:**
```bash
--batch_size 2  # Reduce from 4
--ngf 32        # Reduce generator filters (default 64)
```

---

## Expected Improvements

### Metrics (Approximate)

| Metric | Cold-Start SSVS | ARCADE→SSVS Transfer |
|--------|-----------------|----------------------|
| Training time | ~2-3 weeks | ~10-14 days |
| Convergence epoch | ~100 | ~30-50 |
| Vessel Dice | 0.72-0.78 | 0.78-0.85 |
| Sensitivity | 0.75 | 0.80-0.88 |
| False positives | Higher | Lower |

### Qualitative Signs

- Cleaner vessel boundaries
- Fewer spurious segmentations (noise reduction)
- Better handling of thin vessels
- Faster training convergence

---

## Advanced: Multi-Task Learning (Vessel + Stenosis)

To learn **both** vessel and stenosis simultaneously on ARCADE:

```bash
# Modify Stage 1 to learn both
python train.py \
  --dataroot ./datasets/arcade \
  --name arcade_pretrain_multitask \
  --model arcade_supervision \
  --dataset_mode arcade \
  --arcade_mask_type both \  # Learn both vessel AND stenosis
  --lambda_vessel 1.0 \
  --lambda_stenosis 0.5 \
  --n_epochs 50 \
  ...
```

Then finetune on SSVS as before. This enables:
- Learning shared vessel/stenosis features
- Potential for multi-task inference downstream

---

## Summary

1. ✓ Verify ARCADE data structure
2. ✓ Run Stage 1: `python train.py --model arcade_supervision --dataset_mode arcade ...`
3. ✓ Monitor training until convergence (~50 epochs)
4. ✓ Copy weights to SSVS finetuning checkpoint
5. ✓ Run Stage 2: `python train.py --model usseg --continue_train ...`
6. ✓ Select best checkpoint based on validation loss
7. ✓ Evaluate on held-out test set

**Expected Timeline:**
- Stage 1 (ARCADE pretraining): 3-5 days
- Stage 2 (SSVS finetuning): 1-2 weeks  
- **Total: 2-3 weeks** vs. 2-3 weeks for cold-start (but better quality!)

