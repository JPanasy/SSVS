$ErrorActionPreference = "Continue"

$python = "C:\Users\jkp10\anaconda3\envs\ssv_modern_py310\python.exe"
$name = "arcade_unet_ngf32_scheduled_boundary_lr1e4"
$logDir = "checkpoints\training_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

& $python train.py `
  --dataroot ./datasets/arcade `
  --name $name `
  --allow_existing_run `
  --continue_train `
  --epoch_count 3 `
  --model arcade_supervision `
  --dataset_mode arcade `
  --netG unet_512 `
  --ngf 32 `
  --gpu_ids 0 `
  --lr 0.0001 `
  --seg_loss scheduled_hybrid_boundary `
  --use_boundary_loss `
  --boundary_map_dir ./datasets/arcade/distance_maps `
  --tversky_alpha 0.7 `
  --tversky_beta 0.3 `
  --regional_warmup_epochs 20 `
  --structural_warmup_epochs 50 `
  --boundary_ramp_epochs 50 `
  --lambda_focal 1.0 `
  --lambda_tversky 1.0 `
  --lambda_cldice 0.2 `
  --lambda_bce_final 0.5 `
  --lambda_cldice_final 0.5 `
  --lambda_boundary_final 0.5 `
  --n_epochs 100 `
  --n_epochs_decay 0 `
  --save_epoch_freq 5 `
  --save_latest_freq 1000 `
  --display_id -1 `
  *> (Join-Path $logDir "$name.combined.log")
