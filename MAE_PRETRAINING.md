# MAE Pretraining

This repo now supports masked-autoencoder pretraining as a separate stage from
the supervised ARCADE vessel-mask baseline.

Use this when the goal is to learn reusable vessel image features before a
downstream classifier.

## Architectures

- `gl_mae`: compact global-local masked autoencoder.
- `swin_mae`: Swin Transformer encoder with a reconstruction decoder.

Both use stochastic gradient descent:

- optimizer: `torch.optim.SGD`
- options: `--lr`, `--sgd_momentum`, `--weight_decay`

## Quick Smoke Test

```powershell
C:\Users\jkp10\anaconda3\envs\ssv_modern_py310\python.exe train.py `
  --dataroot ./datasets/arcade `
  --name gl_mae_smoke `
  --model mae_pretrain `
  --dataset_mode arcade `
  --mae_arch gl_mae `
  --n_epochs 1 `
  --n_epochs_decay 0 `
  --batch_size 1 `
  --gpu_ids -1 `
  --max_dataset_size 1 `
  --load_size 128 `
  --crop_size 128 `
  --display_id -1 `
  --no_html
```

## Train GL-MAE

```powershell
C:\Users\jkp10\anaconda3\envs\ssv_modern_py310\python.exe train.py `
  --dataroot ./datasets/arcade `
  --name gl_mae_pretrain `
  --model mae_pretrain `
  --dataset_mode arcade `
  --mae_arch gl_mae `
  --n_epochs 100 `
  --n_epochs_decay 0 `
  --batch_size 8 `
  --gpu_ids 0 `
  --load_size 512 `
  --crop_size 512 `
  --mae_mask_ratio 0.6 `
  --mae_patch_size 16 `
  --mae_local_weight 0.5 `
  --lr 0.01 `
  --sgd_momentum 0.9 `
  --weight_decay 0.05 `
  --print_freq 10 `
  --save_latest_freq 500 `
  --save_epoch_freq 10
```

## Train Swin-MAE

```powershell
C:\Users\jkp10\anaconda3\envs\ssv_modern_py310\python.exe train.py `
  --dataroot ./datasets/arcade `
  --name swin_mae_pretrain `
  --model mae_pretrain `
  --dataset_mode arcade `
  --mae_arch swin_mae `
  --n_epochs 100 `
  --n_epochs_decay 0 `
  --batch_size 2 `
  --gpu_ids 0 `
  --load_size 512 `
  --crop_size 512 `
  --mae_mask_ratio 0.6 `
  --mae_patch_size 16 `
  --mae_local_weight 0.5 `
  --lr 0.01 `
  --sgd_momentum 0.9 `
  --weight_decay 0.05 `
  --print_freq 10 `
  --save_latest_freq 500 `
  --save_epoch_freq 10
```

## Outputs

Checkpoints are saved as:

```text
checkpoints/<run_name>/latest_net_MAE.pth
```

These checkpoints are pretraining feature checkpoints. The next production step
is adding a classifier that loads the MAE encoder and fine-tunes it on vessel
classification labels.
