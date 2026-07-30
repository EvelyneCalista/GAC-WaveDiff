"""
Train a diffusion model to generate images.
"""
import os
import sys
import argparse
import torch as th
import random
import numpy as np
import shutil

sys.path.append("..")
sys.path.append(".")

from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.lidcloader import LIDCVolumes
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    parse_image_size,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            "config",
            "\n".join([f"--{k}={repr(v)} <br/>" for k, v in vars(args).items()]),
        )
        print(
            f"[TENSORBOARD] Using Tensorboard with logdir = {summary_writer.get_logdir()}"
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    dist_util.setup_dist(devices=args.devices)

    logdir_files = os.path.join(summary_writer.get_logdir(), "files/")
    if not os.path.exists(logdir_files):
        os.makedirs(logdir_files)
    shutil.copytree('./scripts/', os.path.join(logdir_files, 'scripts'))
    shutil.copytree('./guided_diffusion/', os.path.join(logdir_files, 'guided_diffusion'))
    shutil.copyfile('./run.sh', os.path.join(logdir_files, 'run.sh'))

    logger.log("[INFO] Creating model and diffusion...")
    logger.log("[ARGS] ", args)
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)
    if args.pretrained_checkpoint:
        logger.log(
            f"[PRETRAINED] Loading weights from "
            f"{args.pretrained_checkpoint}"
        )

        pretrained_state = dist_util.load_state_dict(
            args.pretrained_checkpoint,
            map_location="cpu",
        )

        first_conv_key = "input_blocks.0.0.weight"
        current_state = model.state_dict()

        old_weight = pretrained_state[first_conv_key]
        new_weight = current_state[first_conv_key]

        if old_weight.shape == new_weight.shape:
            pass

        elif (
            old_weight.shape[0] == new_weight.shape[0]
            and old_weight.shape[2:] == new_weight.shape[2:]
            and old_weight.shape[1] < new_weight.shape[1]
        ):
            # Model has extra input channels (e.g. new conditioning channels added on
            # top of an already-trained checkpoint). Keep the old channels' weights
            # as-is and zero-init the new ones, so the model's behavior is initially
            # identical to the checkpoint -- the new channels only start influencing
            # the output once fine-tuning teaches the network to use them.
            old_ch = old_weight.shape[1]
            adapted_weight = th.zeros_like(new_weight)
            adapted_weight[:, :old_ch] = old_weight
            pretrained_state[first_conv_key] = adapted_weight

            logger.log(
                f"[PRETRAINED] Adapted input convolution "
                f"{old_ch} -> {new_weight.shape[1]}; "
                f"new channels' weights initialized to zero."
            )

        else:
            raise ValueError(
                f"Incompatible input convolution: "
                f"checkpoint={tuple(old_weight.shape)}, "
                f"model={tuple(new_weight.shape)}"
            )

        incompatible = model.load_state_dict(
            pretrained_state,
            strict=False,
        )

        logger.log(
            f"[PRETRAINED] Missing keys: {incompatible.missing_keys}"
        )
        logger.log(
            f"[PRETRAINED] Unexpected keys: "
            f"{incompatible.unexpected_keys}"
        )
    print(
        "[MODEL] Number of trainable parameters: {}".format(
            np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
        )
    )
    model.to(
        dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev()
    )  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=args.diffusion_steps
    )
    
    logger.log('model: ', model)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log(f'number of model parameters: {pytorch_total_params}')

    logger.log("[LOGGER] Creating data loader...")

    val_datal = None

    if args.dataset == "brats3d":
        image_size = parse_image_size(args.image_size)
        assert all(s % 2 == 0 for s in image_size)
        print(args.data_dir)
        ds = BRATSVolumes(
            args.data_dir,
            # Precomputed mask-healthy-000N augmentation: only for the training split. Each
            # patient's original pair plus its augmented -0001..-0009 variants (whichever
            # exist on disk) are all included as separate samples; see bratsloader.py.
            use_augmented=True,
            # Rotation augmentation: only for the training split. Each __getitem__ call
            # independently rolls whether to rotate (rotation_prob) and, if so, samples a
            # random small 3D rotation up to rotation_max_angle degrees; see bratsloader.py
            # for how the coordinate-channel volume is rotated along with the anatomy.
            # rotation_augment=args.rotation_aug,
            # rotation_max_angle=args.rotation_max_angle,
            # rotation_prob=args.rotation_prob,
        )

        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

        if args.val_data_dir:
            # use_augmented=False (and rotation_augment left at its default False) so
            # validation loss reflects un-augmented performance, not the easier/harder
            # augmented task.
            val_ds = BRATSVolumes(args.val_data_dir, use_augmented=False)
            val_datal = th.utils.data.DataLoader(
                val_ds,
                batch_size=1,
                num_workers=args.num_workers,
                shuffle=False,
            )

    if args.dataset == "lidc-idri":
        assert args.image_size in [64, 128, 256]
        print(args.data_dir)
        ds = LIDCVolumes(
            args.data_dir,
            test_flag=False,
            normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
            mode="train",
            img_size=args.image_size,
        )

        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

    print(args.resume_checkpoint)
    logger.log("[TRAINING] Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        val_data=val_datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        val_interval=args.val_interval,
        val_batches=args.val_batches,
        mode="default",
        use_sdt=args.use_sdt,
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        val_data_dir="",
        val_interval=500,
        val_batches=4,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        beta_min=0.1,
        beta_max=20.0,
        ema_rate="0.999,0.9999",
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        resume_step=0,
        pretrained_checkpoint="",
        use_sdt=True,  # set False to train/fine-tune without the SDT channel, e.g.
        # for a control run against in_channels=27 with the same dataset/pipeline.
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset="brats3d",
        use_tensorboard=True,
        tensorboard_path="",  # set path to existing logdir for resuming
        devices=[0],
        dims=3,  # 2 for 2d images, 3 for 3d volumes
        learn_sigma=False,
        num_groups=29,
        channel_mult="1,2,2,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode="default",
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        use_wgupdown=False,
        # Rotation data augmentation (BRATSVolumes only, training split only -- see
        # bratsloader.py). Disabled by default so existing training commands are unaffected
        # unless --rotation_aug=True is passed explicitly.
        rotation_aug=False,
        rotation_max_angle=15.0,  # degrees, sampled independently per axis in [-x, x]
        rotation_prob=0.5,  # probability any rotation is applied to a given sample
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
