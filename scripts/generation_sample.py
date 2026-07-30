"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
import random

sys.path.append(".")
import numpy as np
import math
import time
import torch as th
import torch.distributed as dist
import nibabel as nib
import pathlib
import warnings
from scipy.ndimage import distance_transform_edt
from datetime import datetime
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BraTSVolumesTest
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    parse_image_size,
)
#from diffusion import get_time_schedule, Posterior_Coefficients, \
#    sample_from_model_test

from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

from eval import create_submission_adapted
from eval import eval_sam

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / (pred + targs).sum()

def to_range_0_1(x):
    return (x + 1.) / 2.

def axis_window_starts(length, window, stride):
    if window > length:
        raise ValueError(
            f"Window {window} exceeds volume dimension {length}"
        )

    starts = list(range(0, length - window + 1, stride))

    final_start = length - window
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    return starts


def sliding_window_starts(volume_shape, window_shape, stride):
    starts_per_axis = [
        axis_window_starts(length, window, stride)
        for length, window in zip(volume_shape, window_shape)
    ]

    for s0 in starts_per_axis[0]:
        for s1 in starts_per_axis[1]:
            for s2 in starts_per_axis[2]:
                yield s0, s1, s2


def image_to_wavelet(image, dwt):
    """
    Convert [B, 1, D, H, W] images to normalized Haar coefficients.
    """
    return th.cat(dwt(image), dim=1) / math.sqrt(8.0)


def wavelet_to_image(coefficients, idwt):
    """
    Convert normalized [B, 8, D/2, H/2, W/2] Haar coefficients to images.
    """
    scaled = coefficients * math.sqrt(8.0)
    return idwt(
        scaled[:, 0:1],
        scaled[:, 1:2],
        scaled[:, 2:3],
        scaled[:, 3:4],
        scaled[:, 4:5],
        scaled[:, 5:6],
        scaled[:, 6:7],
        scaled[:, 7:8],
    )


def make_blending_weight(window_shape, device):
    axis_weights = [
        th.hann_window(
            size,
            periodic=False,
            device=device,
            dtype=th.float32,
        ).clamp_min(1e-3)
        for size in window_shape
    ]

    weight = (
        axis_weights[0][:, None, None]
        * axis_weights[1][None, :, None]
        * axis_weights[2][None, None, :]
    )

    return weight / weight.max()


def sample_per_step_consensus(
    *,
    model,
    diffusion,
    dwt,
    idwt,
    full_voided,
    full_mask,
    target_mask,
    sdt_full,
    window_shape,
    stride,
    blend_weight,
    args,
    device,
    iteration,
):
    """
    Fuse overlapping 3D patch predictions after every reverse-diffusion step.

    Each reverse state is represented by one global wavelet tensor. Window
    predictions are inverse-transformed, Hann-fused in image space, and then
    transformed back to wavelet space before the next reverse step.
    """
    if args.num_seeds != 1:
        raise ValueError(
            "Per-step consensus currently requires --num_seeds=1"
        )
    if args.tta_flip:
        raise ValueError(
            "Disable TTA while validating per-step consensus"
        )
    if args.sampling_noise_scale != 0.0:
        raise ValueError(
            "Per-step consensus currently requires "
            "--sampling_noise_scale=0.0"
        )
    if args.sampling_steps != diffusion.num_timesteps:
        raise ValueError(
            f"sampling_steps={args.sampling_steps}, but diffusion has "
            f"{diffusion.num_timesteps} timesteps"
        )

    if len(full_voided.shape) != 3 or len(window_shape) != 3:
        raise ValueError(
            "Per-step consensus expects 3D volumes and 3D windows"
        )

    D, H, W = full_voided.shape
    d, h, w = window_shape

    if any(size % 2 != 0 for size in (D, H, W, d, h, w)):
        raise ValueError(
            "Haar wavelet consensus requires even volume and window sizes"
        )

    active_windows = []
    for s0, s1, s2 in sliding_window_starts(
        full_voided.shape,
        window_shape,
        stride,
    ):
        if any(start % 2 != 0 for start in (s0, s1, s2)):
            raise ValueError(
                "Haar wavelet consensus requires even window starts; "
                f"received {(s0, s1, s2)}"
            )

        spatial_slice = (
            slice(s0, s0 + d),
            slice(s1, s1 + h),
            slice(s2, s2 + w),
        )

        if not target_mask[spatial_slice].any():
            continue

        latent_slice = (
            slice(s0 // 2, s0 // 2 + d // 2),
            slice(s1 // 2, s1 // 2 + h // 2),
            slice(s2 // 2, s2 // 2 + w // 2),
        )

        active_windows.append(
            {
                "spatial_slice": spatial_slice,
                "latent_slice": latent_slice,
                "crop_start": np.asarray(
                    [[s0, s1, s2]],
                    dtype=np.float32,
                ),
            }
        )

    if not active_windows:
        raise RuntimeError(
            "No sliding window contains target-mask voxels"
        )

    logger.log(
        f"[CONSENSUS] active windows: {len(active_windows)}"
    )

    case_seed = args.seed + iteration * 1_000_000
    th.manual_seed(case_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(case_seed)

    global_xt = th.randn(
        1,
        8,
        D // 2,
        H // 2,
        W // 2,
        device=device,
    )

    final_prediction = None
    final_covered = None

    with th.no_grad():
        for timestep in reversed(range(diffusion.num_timesteps)):
            logger.log(f"[CONSENSUS] timestep: {timestep}")

            t_batch = th.full(
                (1,),
                timestep,
                device=device,
                dtype=th.long,
            )
            step_sum = th.zeros_like(full_voided)
            step_weight = th.zeros_like(full_voided)

            for record in active_windows:
                spatial_slice = record["spatial_slice"]
                latent_slice = record["latent_slice"]

                patch_xt = global_xt[
                    (slice(None), slice(None)) + latent_slice
                ].contiguous()

                expected_shape = (
                    1,
                    8,
                    d // 2,
                    h // 2,
                    w // 2,
                )
                if tuple(patch_xt.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected latent crop shape "
                        f"{tuple(patch_xt.shape)}; expected {expected_shape}"
                    )

                patch_voided = (
                    full_voided[spatial_slice]
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                patch_mask = (
                    full_mask[spatial_slice]
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                patch_sdt = (
                    sdt_full[spatial_slice]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    if sdt_full is not None
                    else None
                )

                input_masked_dwt = image_to_wavelet(
                    patch_voided,
                    dwt,
                ).clamp(-1, 1)
                mask_dwt = image_to_wavelet(
                    patch_mask,
                    dwt,
                ).clamp(-1, 1)

                step_output = diffusion.p_sample(
                    model=model,
                    x=patch_xt,
                    t=t_batch,
                    input_masked=input_masked_dwt,
                    mask=mask_dwt,
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                    crop_start=record["crop_start"],
                    flip_axes=(),
                    sdt=patch_sdt,
                    sampling_noise_scale=args.sampling_noise_scale,
                )

                patch_next_image = wavelet_to_image(
                    step_output["sample"],
                    idwt,
                )[0, 0]

                if timestep == 0:
                    patch_next_image = patch_next_image.clamp(-1, 1)

                # Fuse the complete patch. The lesion mask is applied only to
                # the final reconstructed volume, not during consensus.
                step_sum[spatial_slice] += (
                    patch_next_image * blend_weight
                )
                step_weight[spatial_slice] += blend_weight

            covered = step_weight > 0
            missing_target = target_mask & ~covered
            if missing_target.any():
                raise RuntimeError(
                    f"Consensus missed "
                    f"{missing_target.sum().item()} target voxels at "
                    f"timestep {timestep}"
                )

            # Retain the previous global state outside the union of active
            # windows. Division is restricted to positive weights so the
            # smallest Hann weights are not artificially clamped.
            global_next_image = wavelet_to_image(
                global_xt,
                idwt,
            )[0, 0]
            global_next_image[covered] = (
                step_sum[covered] / step_weight[covered]
            )

            if not th.isfinite(global_next_image[covered]).all():
                raise RuntimeError(
                    f"NaN or Inf detected at timestep {timestep}"
                )

            if timestep > 0:
                global_xt = image_to_wavelet(
                    global_next_image.unsqueeze(0).unsqueeze(0),
                    dwt,
                )
            else:
                final_prediction = global_next_image.clamp(-1, 1)
                final_covered = covered

    if final_prediction is None or final_covered is None:
        raise RuntimeError(
            "Consensus did not produce a final prediction"
        )

    return final_prediction, final_covered


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    save_dir = os.path.join(args.output_dir, args.model_path.split('/')[-1].split('.')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.configure(dir=save_dir)

    start_time = time.time()

    logger.log("[INFO] creating model and diffusion...")
    logger.log("[ARGS] ", args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("[INFO] load model from: {}".format(args.model_path))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    logger.log("[MODEL] ", model)
    model.to(
        dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev()
    )  # allow for 2 devices

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    dataset = BraTSVolumesTest(folder1=args.data_dir)
    test_sampler = th.utils.data.SequentialSampler(dataset)

    data_loader = th.utils.data.DataLoader(dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True,
                                           sampler=test_sampler,
                                           drop_last=True)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    num_sam = 0

    start_actual_sampling = time.time()

    # for iteration, (stacked_images, voided_image_full, starts, ends, labeled_masks, file_name) in enumerate(
    #         data_loader):
    for iteration, (
        stacked_images,
        voided_image_full,
        full_mask,
        starts,
        ends,
        labeled_masks,
        file_name,
    ) in enumerate(data_loader):
        logger.log("sampling file ", str(file_name))
        logger.log("    sampling start ", datetime.now())

        # th.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)
        # print(f"reseeded (in for loop) to {seed}")

        # seed += 1

        device = dist_util.dev()

        window_shape = parse_image_size(args.image_size)
        stride = args.window_stride

        full_voided = voided_image_full[0].to(
            device=device,
            dtype=th.float32,
        )

        full_mask = full_mask[0].to(
            device=device,
            dtype=th.float32,
        )

        if full_voided.shape != full_mask.shape:
            raise ValueError(
                f"Image and mask shapes differ: "
                f"{full_voided.shape} versus {full_mask.shape}"
            )

        target_mask = full_mask == 1

        if args.use_sdt:
            # Computed ONCE on the full volume (not per-window) so distances reflect the
            # true lesion geometry even when a sliding window only captures part of a
            # large mask -- matching how BRATSVolumes.__getitem__ computes it on the full
            # padded volume before cropping during training. Sliced per-window below the
            # same way full_mask/full_voided are.
            full_mask_np = (full_mask.detach().cpu().numpy() == 1)
            dist_inside = distance_transform_edt(full_mask_np)
            dist_outside = distance_transform_edt(~full_mask_np)
            sdt_full = th.as_tensor(
                (dist_inside - dist_outside).astype(np.float32),
                device=device,
            )
        else:
            sdt_full = None

        blend_weight = make_blending_weight(
            window_shape,
            device,
        )

        if args.per_step_consensus:
            averaged_prediction, covered_mask = (
                sample_per_step_consensus(
                    model=model,
                    diffusion=diffusion,
                    dwt=dwt,
                    idwt=idwt,
                    full_voided=full_voided,
                    full_mask=full_mask,
                    target_mask=target_mask,
                    sdt_full=sdt_full,
                    window_shape=window_shape,
                    stride=stride,
                    blend_weight=blend_weight,
                    args=args,
                    device=device,
                    iteration=iteration,
                )
            )
            baseline_windows = ()
        else:
            prediction_sum = th.zeros_like(full_voided)
            weight_sum = th.zeros_like(full_voided)
            baseline_windows = enumerate(
                sliding_window_starts(
                    full_voided.shape,
                    window_shape,
                    stride,
                )
            )

        with th.no_grad():
            for window_index, (s0, s1, s2) in baseline_windows:
                d, h, w = window_shape

                spatial_slice = (
                    slice(s0, s0 + d),
                    slice(s1, s1 + h),
                    slice(s2, s2 + w),
                )

                patch_mask_original = full_mask[spatial_slice]
                patch_binary_mask = patch_mask_original == 1

                # Skip windows that contain no missing voxels.
                if not patch_binary_mask.any():
                    continue

                patch_voided_base = (
                    full_voided[spatial_slice]
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                patch_mask_base = (
                    patch_mask_original
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                if args.use_sdt:
                    patch_sdt_base = (
                        sdt_full[spatial_slice]
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                else:
                    patch_sdt_base = None

                crop_start = np.asarray(
                    [[s0, s1, s2]],
                    dtype=np.float32,
                )

                # Each entry is either None (identity view) or a spatial axis
                # (0/1/2, matching (D, H, W)) to flip for test-time augmentation.
                # views stays [None] -- i.e. bit-for-bit the old behavior and seed
                # numbering -- unless --tta_flip is set.
                views = [None]
                if args.tta_flip:
                    views.append(args.tta_flip_axis)

                reconstructed_samples = []

                for view_idx, view_axis in enumerate(views):
                    if view_axis is None:
                        patch_voided = patch_voided_base
                        patch_mask = patch_mask_base
                        patch_sdt = patch_sdt_base
                        flip_axes = ()
                    else:
                        # +2 to skip the (batch, channel) dims of the (1, 1, D, H, W)
                        # tensors when flipping.
                        patch_voided = th.flip(patch_voided_base, dims=[view_axis + 2])
                        patch_mask = th.flip(patch_mask_base, dims=[view_axis + 2])
                        # SDT is derived directly from the mask, so flipping the
                        # precomputed field the same way keeps it consistent --
                        # no crop_start-style rebuild needed (see p_mean_variance).
                        patch_sdt = (
                            th.flip(patch_sdt_base, dims=[view_axis + 2])
                            if patch_sdt_base is not None
                            else None
                        )
                        flip_axes = (view_axis,)

                    bands = dwt(patch_voided)
                    input_masked_dwt = th.cat(bands, dim=1)
                    input_masked_dwt /= math.sqrt(8.0)
                    input_masked_dwt = input_masked_dwt.clamp(-1, 1)

                    mask_bands = dwt(patch_mask)
                    mask_dwt = th.cat(mask_bands, dim=1)
                    mask_dwt /= math.sqrt(8.0)
                    mask_dwt = mask_dwt.clamp(-1, 1)

                    for seed_index in range(args.num_seeds):
                        # member_seed = (
                        #     args.seed
                        #     + iteration * 100000
                        #     + window_index * args.num_seeds * len(views)
                        #     + view_idx * args.num_seeds
                        #     + seed_index
                        # )
                        member_seed = (
                            args.seed
                            + iteration * 1_000_000
                            + window_index * 10_000
                            + view_idx * 100
                            + seed_index
                        )

                        th.manual_seed(member_seed)
                        if th.cuda.is_available():
                            th.cuda.manual_seed_all(member_seed)

                        noise = th.randn(
                            1,
                            8,
                            d // 2,
                            h // 2,
                            w // 2,
                            device=device,
                        )

                        sample = diffusion.p_sample_loop(
                            t=args.sampling_steps,
                            model=model,
                            shape=noise.shape,
                            noise=noise,
                            input_masked=input_masked_dwt,
                            mask=mask_dwt,
                            clip_denoised=args.clip_denoised,
                            model_kwargs={},
                            crop_start=crop_start,
                            flip_axes=flip_axes,
                            sdt=patch_sdt,sampling_noise_scale=args.sampling_noise_scale,
                        )

                        sample = sample * math.sqrt(8.0)

                        patch_prediction = idwt(
                            sample[:, 0:1],
                            sample[:, 1:2],
                            sample[:, 2:3],
                            sample[:, 3:4],
                            sample[:, 4:5],
                            sample[:, 5:6],
                            sample[:, 6:7],
                            sample[:, 7:8],
                        )

                        patch_prediction = patch_prediction[0, 0].clamp(-1, 1)

                        if view_axis is not None:
                            # Undo the input flip so every view lands back in the
                            # same (original) orientation before averaging.
                            patch_prediction = th.flip(
                                patch_prediction, dims=[view_axis]
                            )

                        reconstructed_samples.append(patch_prediction)

                patch_prediction = th.stack(
                    reconstructed_samples,
                    dim=0,
                ).mean(dim=0)

                patch_weight = (
                    blend_weight * patch_binary_mask.float()
                )

                prediction_sum[spatial_slice] += (
                    patch_prediction * patch_weight
                )

                weight_sum[spatial_slice] += patch_weight

        if not args.per_step_consensus:
            covered_mask = weight_sum > 0
            averaged_prediction = th.zeros_like(full_voided)
            averaged_prediction[covered_mask] = (
                prediction_sum[covered_mask]
                / weight_sum[covered_mask]
            )

        missing_mask = target_mask & ~covered_mask

        if missing_mask.any():
            raise RuntimeError(
                f"Sliding windows missed "
                f"{missing_mask.sum().item()} masked voxels"
            )

        reconstructed_full = full_voided.clone()
        reconstructed_full[target_mask] = averaged_prediction[target_mask]

        voided_image_full = (
            reconstructed_full.clamp(-1, 1) + 1.0
        ) / 2.0
        #nib.save(nib.Nifti1Image(np.asarray(voided_image_full.cpu().detach().squeeze()), None),
        #         os.path.join(save_dir, file_name[0].replace('t1n-voided', 't1n-inference')))

        input_name = file_name[0]
        #replace as same as organizer
        case_id = input_name.replace("-t1n-voided.nii.gz", "")

        case_dir = os.path.join(args.data_dir, case_id)
        input_path = os.path.join(case_dir, input_name)
        mask_path = os.path.join(
            case_dir,
            case_id + "-mask.nii.gz",
        )

        # Load original organizer files.
        input_nifti = nib.load(input_path)
        original_voided = input_nifti.get_fdata(dtype=np.float32)
        original_mask = nib.load(mask_path).get_fdata() == 1

        # Organizer normalization uses the maximum nonnegative input intensity.
        # nonnegative_input = np.maximum(original_voided, 0)
        # max_v = float(nonnegative_input.max())

        # if max_v <= 0:
        #     raise ValueError(
        #         f"Invalid max_v={max_v} for {input_name}"
        #     )
        # Match the percentile clipping used during preprocessing.
        clip_low = float(np.quantile(original_voided, 0.005))
        clip_high = float(np.quantile(original_voided, 0.995))

        if clip_high <= clip_low:
            raise ValueError(
                f"Invalid intensity range for {input_name}: "
                f"clip_low={clip_low}, clip_high={clip_high}"
            )
        # Remove WDM3D padding: 256³ -> 240×240×155.
        prediction_01 = (
            voided_image_full
            .detach()
            .cpu()
            .numpy()[8:-8, 8:-8, 50:-51]
            .astype(np.float32)
        )

        if prediction_01.shape != original_voided.shape:
            raise ValueError(
                f"Prediction shape {prediction_01.shape} does not match "
                f"input shape {original_voided.shape}"
            )

        # Restore organizer MRI intensity scale.
        # prediction_original = prediction_01 * max_v
        prediction_original = (
            prediction_01 * (clip_high - clip_low)
            + clip_low
        )

        # Preserve the original image outside the mask.
        result = original_voided.copy()
        result[original_mask] = prediction_original[original_mask]

        submission_dir = os.path.join(
            save_dir,
            "brats_submission",
        )
        os.makedirs(submission_dir, exist_ok=True)

        output_name = input_name.replace(
            "t1n-voided",
            "t1n-inference",
        )
        output_path = os.path.join(
            submission_dir,
            output_name,
        )

        output_header = input_nifti.header.copy()
        output_header.set_data_dtype(np.float32)

        nib.save(
            nib.Nifti1Image(
                result,
                input_nifti.affine,
                output_header,
            ),
            output_path,
        )

        print(
            "Saved:",
            output_path,
            "shape:",
            result.shape,
            "range:",
            float(result.min()),
            float(result.max()),
            # "max_v:",
            # max_v,
            "clip range:",
            (clip_low, clip_high),
        )
        num_sam += 1

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    logger.info(f"Average time per sample (including loading): {elapsed_time / num_sam:.6f} seconds")

    elapsed_time_sampling_only = end_time - start_actual_sampling
    print(f"Elapsed time: {elapsed_time_sampling_only:.6f} seconds")
    logger.info(f"Average time per sample (sampling only): {elapsed_time_sampling_only / num_sam:.6f} seconds")

    print('preparing samples for evaluation ...')
    # create_submission_adapted.adapt(input_data=args.data_dir, samples_dir=save_dir, adapted_samples_dir=save_dir)
    adapted_dir = os.path.join(save_dir, "brats_submission")
    os.makedirs(adapted_dir, exist_ok=True)
    #create_submission_adapted.adapt(input_data=args.data_dir, samples_dir=save_dir, adapted_samples_dir=adapted_dir)
    print("Samples saved in:",os.path.join(save_dir, "brats_submission"),)
    #print('samples ready for evaluation, saved in ', save_dir)
    #eval_sam.eval_adapted_samples(
    #    dataset_path_eval=args.data_dir,
    #    solutionFilePaths_gt=args.data_dir,
    #    resultsFolder_dir=save_dir)


def create_argparser():
    defaults = dict(
        seed=0,
        num_seeds=4,
        data_dir="",
        data_mode="validation",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir="./results",
        mode="default",
        window_stride=64,
        sampling_noise_scale=1.0,
        per_step_consensus=False,
        tta_flip=False,  # average an extra flipped-and-flipped-back view per window
        tta_flip_axis=0,  # spatial axis (0/1/2 of D,H,W) to flip -- verify this is
        # actually left-right for your data before trusting it (see run_inference.sh)
        use_sdt=False,  # append the signed-distance-to-mask-boundary conditioning
        # channel; must match how the model being loaded was trained (in_channels
        # 27 vs 28) or the first conv layer's shape won't match the checkpoint.
        renormalize=False,
        image_size="256",
        half_res_crop=False,
        concat_coords=False,  # unused leftover flag; the real coordinate-channel wiring
        # lives in bratsloader.py/gaussian_diffusion.py and is driven by --in_channels,
        # not by this flag.
    )
    defaults.update(
        {k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults}
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

