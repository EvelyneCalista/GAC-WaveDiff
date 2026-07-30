# general settings
GPU=0;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
MODE='sample';            # train vs sample
DATA_MODE="test"        # train, test, validation data
MODEL='ours_unet_128';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'ours_unet_128' ]]; then
  echo "Ours (U-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,2,4;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
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
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  echo "Ours (WavU-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=3;
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "Ours (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE_TRAIN=256;
  IMAGE_SIZE_SAMPLE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=3;
elif [[ $MODEL == 'ours_unet_mamba_Centroid_bbox_192' ]]; then
  echo "Ours (U-Net) 192 x 192 x 160";
  CHANNEL_MULT=1,2,2,2,4;
  IMAGE_SIZE_TRAIN=192,192,160;
  IMAGE_SIZE_SAMPLE=192,192,160;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;   # 224x224x160 體積約為 128^3 的 3.8 倍,記憶體會吃緊,batch_size 建議先降到 1
  USE_MAMBA=True
else
  echo "MODEL TYPE NOT FOUND";
fi

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
--data_dir=/train_2023
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE_TRAIN}
--use_fp16=False
--lr=2e-5
--save_interval=5000
--num_workers=4
--devices=${GPU}
--lr_anneal_steps=120000
"
SAMPLE="
--data_dir=/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation
--data_mode=${DATA_MODE}
--use_sdt=True
--seed=${SEED}
--image_size=${IMAGE_SIZE_SAMPLE}
--use_fp16=False
--devices=${GPU}
--num_samples=1000
--use_ddim=False
--clip_denoised=True
--window_stride=64
--num_seeds=1
--tta_flip=True
--tta_flip_axis=0
--sampling_noise_scale=0.0
--per_step_consensus=False
"
if [[ $MODE == 'train' ]]; then
  python scripts/generation_train.py $TRAIN $COMMON;
else
  python scripts/generation_sample.py $SAMPLE $COMMON --diffusion_steps 2 --sampling_steps 2 --model_path /checkpoints/brats3dimage.pt --output_dir ./brats_CLEAN/ ;
fi
