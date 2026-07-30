# general settings
GPU=0;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
MODE='train';            # train vs sample
DATA_MODE="train"        # train, test, validation data
MODEL='ours_unet_128';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'ours_unet_128' ]]; then
  echo "Ours (U-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,2,4;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=3;
  USE_MAMBA=False;
elif [[ $MODEL == 'ours_unet_mamba_128' ]]; then
  echo "ours_unet_mamba_128 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,2,4;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=3;
  USE_MAMBA=True;
elif [[ $MODEL == 'ours_unet_256' ]]; then
  echo "Ours (U-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE_TRAIN=256;
  IMAGE_SIZE_SAMPLE=256;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=3;
  USE_MAMBA=False;
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  echo "Ours (WavU-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=3;
  USE_MAMBA=False;
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "Ours (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE_TRAIN=256;
  IMAGE_SIZE_SAMPLE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=3;
  USE_MAMBA=False;
elif [[ $MODEL == 'ours_unet_mamba_Centroid_bbox_192' ]]; then
  echo "ours_unet_mamba 192x192x160"
  CHANNEL_MULT=1,2,2,2,4
  IMAGE_SIZE_TRAIN=192,192,160
  IMAGE_SIZE_SAMPLE=192,192,160
  ADDITIVE_SKIP=True
  USE_FREQ=False
  BATCH_SIZE=3
  USE_MAMBA=True
else
  echo "MODEL TYPE NOT FOUND";
fi

# in_channels=28: 24 (x_t + voided-MRI-dwt + mask-dwt) + 3 coordinate channels (x,y,z) + 1 mask SDT channel
COMMON="
--beta_min=0.1
--beta_max=20.0
--dataset=brats3d
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=4
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--use_checkpoint=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=2
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=28
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
--use_wgupdown=False
--use_mamba=${USE_MAMBA}
"
TRAIN="
--data_dir=/home/evelyne/Documents/inpainting/2023split_1200_51/train_1200
--val_data_dir=/home/evelyne/Documents/inpainting/2023split_1200_51/val_51
--val_interval=500
--val_batches=4
--resume_checkpoint=
--resume_step=0
--use_sdt=True
--image_size=${IMAGE_SIZE_TRAIN}
--use_fp16=False
--lr=2e-5
--ema_rate=0.999,0.9999
--save_interval=5000
--num_workers=4
--devices=${GPU}
--lr_anneal_steps=120000
--rotation_aug=False
--rotation_max_angle=5.0
--rotation_prob=0.5
"
SAMPLE="
--data_dir=/home/evelyne/Documents/inpainting/2023split_1200_51/val_51
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE_SAMPLE}
--use_fp16=False
--devices=${GPU}
--num_samples=1000
--use_ddim=False
--clip_denoised=True
"
if [[ $MODE == 'train' ]]; then
  python scripts/generation_train.py $TRAIN $COMMON;
else
  python scripts/generation_sample.py $SAMPLE $COMMON --diffusion_steps 2 --sampling_steps 2 --model_path /path/to/checkpoint.pt --output_dir ./sampling_output/ ;
fi
