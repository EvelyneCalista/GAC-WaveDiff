# import math
# import os
# import os.path
# import warnings

# import nibabel as nib
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data
# from skimage.measure import label
# def _crop_start_centroid_bbox(centroid, bbox_min, bbox_max, crop_size=(224, 224, 160), volume_size=256, margin=16):
#     # `start` (returned below) is a voxel offset into this volume_size^3 frame. It is also
#     # the origin used by the coordinate-channel feature (GaussianDiffusion.
#     # build_coordinate_channels() in gaussian_diffusion.py), which assumes GLOBAL_VOLUME_SIZE
#     # == volume_size == 256; keep the two in sync if this ever changes.
#     start = np.empty(3, dtype=int)
#     for i in range(3):
#         b_min = max(int(bbox_min[i]) - margin, 0)
#         b_max = min(int(bbox_max[i]) + margin, volume_size - 1)

#         if b_max - b_min + 1 > crop_size[i]:
#             raise ValueError(
#                 f"Mask with {margin}-voxel margin spans {b_max - b_min + 1} voxels "
#                 f"on axis {i}, exceeding the crop window size {crop_size[i]}."
#             )

#         s_lower = max(0, b_max - crop_size[i] + 1)
#         s_upper = min(b_min, volume_size - crop_size[i])

#         s_preferred = int(centroid[i]) - crop_size[i] // 2
#         start[i] = int(np.clip(s_preferred, s_lower, s_upper))
#     return start


# # CROP_SHAPE = (192, 192, 160)
# CROP_SHAPE = (128, 128, 128)

# # Fill value used for every channel (image, mask, coordinates) wherever a rotation warps
# # in a region that has no source voxel (i.e. the corners of the crop). Matches the -1
# # background/"no lesion" constant already used by the np.pad(..., constant_values=-1) calls
# # above, so rotated-in corners look like the same padding the rest of the pipeline expects.
# ROTATION_BACKGROUND = -1.0


# def _random_rotation_theta(max_angle_deg):
#     """
#     Sample a random small 3D rotation (independent Euler angles about x/y/z, each in
#     [-max_angle_deg, max_angle_deg]). A single combined rotation is used (instead of three
#     sequential 2D rotations) so the volume is resampled only once, avoiding the extra
#     interpolation blur that chaining separate axis rotations would add.

#     Returns (theta, rotation):
#     - `rotation` is the 3x3 matrix in the tensor's own (D, H, W) axis order -- this is the
#       one to reason about anywhere else in this file (e.g. the margin math in
#       _rotation_source_shape), including printing/debugging.
#     - `theta` is the (3, 4) affine matrix to actually pass to
#       torch.nn.functional.affine_grid/_rotate_volume. IMPORTANT: PyTorch's grid_sample for
#       5D volumetric input reads theta's row/col index 0 as the tensor's *last* spatial axis
#       (W) and index 2 as the *first* (D) -- reversed from the tensor's own (D, H, W) axis
#       order (verified empirically; not documented clearly). `rotation` is therefore permuted
#       (both rows and columns reversed) before being embedded into theta, so that the
#       rotation PyTorch actually performs -- expressed back in (D, H, W) terms -- matches
#       `rotation`, not its axis-reversed twin.
#     """
#     rx, ry, rz = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg, size=3))
#     Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
#     Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
#     Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
#     rotation = (Rz @ Ry @ Rx).astype(np.float32)  # (D, H, W)-ordered

#     pytorch_rotation = rotation[::-1, ::-1].copy()  # axis-reversed for grid_sample's convention
#     theta = np.zeros((3, 4), dtype=np.float32)
#     theta[:, :3] = pytorch_rotation
#     return torch.from_numpy(theta), torch.from_numpy(rotation)


# def _rotate_volume(volume, theta, mode, background):
#     """
#     Resample `volume` (C, D, H, W) with the rotation `theta` about the volume's center.

#     grid_sample() only supports 0-padding for out-of-grid samples, but our pipeline's
#     background/"no lesion" sentinel is -1 everywhere else. We work around this by shifting
#     the data so `background` maps to 0 before resampling (so 0-padding reproduces exactly
#     `background` for the rotated-in corners), then shifting back afterwards.
#     """
#     shifted = (volume - background).unsqueeze(0)  # (1, C, D, H, W); background -> 0
#     grid = F.affine_grid(theta.unsqueeze(0), shifted.shape, align_corners=False)
#     rotated = F.grid_sample(
#         shifted, grid, mode=mode, padding_mode="zeros", align_corners=False
#     )
#     return rotated.squeeze(0) + background


# def _rotation_source_shape(crop_shape, rotation, volume_size):
#     """
#     "Zero information loss" rotation support: compute the minimal axis-aligned box
#     (>= crop_shape) that must be sampled from BEFORE applying `rotation`, so that a
#     centered `crop_shape` region cut out of the *rotated* result is guaranteed to be built
#     entirely from real voxels rather than _rotate_volume()'s background fill.

#     :param rotation: the (D, H, W)-ordered 3x3 rotation matrix returned by
#         _random_rotation_theta() -- NOT the (axis-reversed) theta passed to grid_sample; see
#         that function's docstring for why the two differ.

#     grid_sample() (align_corners=False) maps voxel index i in an axis of size N to
#     normalized coordinate 2*(i - (N-1)/2)/N, i.e. the outermost sampled voxel of a centered
#     window of `crop_shape` voxels sits (crop_shape-1)/2 voxels from that window's center
#     (NOT crop_shape/2 -- off by half a voxel per axis, which matters once amplified by the
#     row-sum below). For output position p measured this way, the source position is
#     rotation @ p; rotation is orthogonal, so for p inside a centered box of half-extents
#     (crop_shape-1)/2, the source coordinate along axis i is bounded by
#     sum_j |rotation_ij| * (crop_shape[j]-1) / 2. This uses the *actual* rotation sampled for
#     this example (not a fixed worst-case angle bound) so the box is as tight as possible --
#     a fixed-angle worst-case bound would either waste extra context (if conservative) or,
#     worse, understate what's needed for the compound 3-axis rotation
#     _random_rotation_theta() applies (verified numerically: chaining independent x/y/z
#     rotations needs more margin than the single-plane cos+sin formula alone suggests).

#     The bound above is an equality in the worst case (a corner voxel of the crop), so it
#     has zero slack against float32 rounding -- verified empirically (300 random-rotation
#     trials) to leak a handful of boundary voxels without some safety margin. SAFETY_MARGIN
#     pads it by 5% (rounded up to whole voxels) so the guarantee holds in practice, not just
#     in exact arithmetic.
#     """
#     SAFETY_MARGIN = 1.05
#     half_crop = (torch.tensor(crop_shape, dtype=torch.float32) - 1.0) / 2.0
#     half_outer = rotation.abs() @ half_crop  # (3,) required half-extent per source axis, (D,H,W)-ordered
#     outer = [int(math.ceil(2.0 * v.item() * SAFETY_MARGIN)) for v in half_outer]
#     outer = [max(o, c) for o, c in zip(outer, crop_shape)]  # never shrink below crop_shape
#     if any(o > volume_size for o in outer):
#         raise ValueError(
#             f"rotation_max_angle is too large for CROP_SHAPE={crop_shape}: this sample's "
#             f"zero-information-loss source box would need shape {tuple(outer)}, which "
#             f"exceeds the padded volume size {volume_size}. Lower rotation_max_angle."
#         )
#     return tuple(outer)


# def _expand_crop_bounds(start, inner_shape, outer_shape, volume_size):
#     """
#     Given `start` (origin of the already lesion-validated inner_shape crop returned by
#     crop_mask_to_center/_crop_start_centroid_bbox), compute the origin of a larger,
#     concentric outer_shape box, clamped to stay inside the volume_size^3 frame.

#     Returns (outer_start, offset, clamped):
#     - `offset` is where the inner crop sits inside the outer box (offset = start -
#       outer_start). Slicing the outer box at [offset : offset + inner_shape] -- even after
#       rotating it -- always reconstructs exactly the start:start+inner_shape window of the
#       *unrotated* volume: clamping only shifts where that window sits inside the outer box,
#       it never changes which voxels belong to the final crop.
#     - `clamped` is True if `start` was too close to the volume edge for the outer box to
#       stay centered without hitting the volume boundary. _rotation_source_shape()'s margin
#       is only guaranteed sufficient on *both* sides of an axis when the box stays centered;
#       once clamped, the margin on the side that got compressed can run short, so a sliver of
#       ROTATION_BACKGROUND may leak into the final crop after all on that side. Zero
#       information loss is not achievable at this crop location for this rotation -- the
#       caller should surface this (see the warning in BRATSVolumes.__getitem__) rather than
#       silently degrade.
#     """
#     outer_start = np.empty(3, dtype=int)
#     offset = np.empty(3, dtype=int)
#     clamped = False
#     for i in range(3):
#         extra = outer_shape[i] - inner_shape[i]
#         candidate = int(start[i] - extra // 2)
#         clipped = int(np.clip(candidate, 0, volume_size - outer_shape[i]))
#         if clipped != candidate:
#             clamped = True
#         outer_start[i] = clipped
#         offset[i] = start[i] - clipped
#     return outer_start, offset, clamped


# class BRATSVolumes(torch.utils.data.Dataset):
#     def __init__(self, folder1, rotation_augment=False, rotation_max_angle=15.0, rotation_prob=0.5):
#         # Collect all patient folders
#         self.patients = sorted([d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
#         self.folder1 = folder1

#         self.labels = [0] * len(self.patients)  # Assuming all images have the same label for now
#         self._shape_printed = False
#         # Rotation augmentation settings (see _random_rotation_theta/_rotate_volume above).
#         # Disabled by default so BraTSVolumesTest/Val-style deterministic use is unaffected;
#         # generation_train.py explicitly turns this on for the training split only.
#         self.rotation_augment = rotation_augment
#         self.rotation_max_angle = rotation_max_angle
#         self.rotation_prob = rotation_prob
#     def __len__(self):
#         return len(self.patients)

#     def __getitem__(self, idx):
#         patient_folder = os.path.join(self.folder1, self.patients[idx])

#         mask_name = self.patients[idx] + '-mask-healthy.nii.gz'
#         t1n_name = self.patients[idx] + '-t1n.nii.gz'
#         mask_file = os.path.join(patient_folder, mask_name)
#         t1n_file = os.path.join(patient_folder, t1n_name)

#         # Load mask
#         mask_image = self.load_data(mask_file)
#         #nib.save(nib.Nifti1Image(np.asarray(mask_image), None), 'mask_loaded.nii.gz')
#         # Set non-one values to minus 1
#         mask_image[mask_image != 1] = -1
#         #nib.save(nib.Nifti1Image(np.asarray(mask_image), None), 'mask_set_to_minus1.nii.gz')
#         # Pad mask
#         mask_image = np.pad(mask_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
#         #nib.save(nib.Nifti1Image(np.asarray(mask_image), None), 'mask_after_pad.nii.gz')
#         # Keep the padded 256^3 array around: crop_mask_to_center() below only *views* it to
#         # get the (lesion-validated) CROP_SHAPE window, but if this sample rotates we also
#         # need a larger, concentric window out of the same padded volume (see will_rotate
#         # branch below).
#         mask_padded = mask_image
#         # Crop to desired region
#         cropped_mask, start, end = self.crop_mask_to_center(mask_padded)
#         #nib.save(nib.Nifti1Image(np.asarray(mask_image), None), 'mask_after_crop.nii.gz')
#         #if mask_image.shape != (128, 128, 128):
#         #    raise ValueError("Generated mask {} must be of shape (128, 128, 128) but is {}!".format(mask_file, mask_image.shape))
#         if cropped_mask.shape != CROP_SHAPE:
#             raise ValueError("Generated mask {} must be of shape {} but is {}!".format(mask_file, CROP_SHAPE, cropped_mask.shape))

#         t1n_image = self.load_data(t1n_file)
#         t1n_image[t1n_image < 0] = 0
#         max_v = float(np.max(t1n_image))
#         if max_v <= 0:
#             raise ValueError(f"Maximum intensity must be positive for {t1n_file}, got {max_v}")
#         t1n_image = 2.0 * (t1n_image / max_v) - 1.0
#         t1n_image = np.pad(t1n_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
#         # Also kept at full 256^3 resolution for the same reason as mask_padded above.
#         t1n_padded = torch.as_tensor(t1n_image, dtype=torch.float32)

#         will_rotate = self.rotation_augment and np.random.rand() < self.rotation_prob
#         if will_rotate:
#             theta, rotation = _random_rotation_theta(self.rotation_max_angle)
#         else:
#             theta, rotation = None, None

#         if will_rotate:
#             # Zero-information-loss rotation: instead of rotating the CROP_SHAPE window in
#             # place (which would fill its corners with ROTATION_BACKGROUND wherever the
#             # rotation warps in unknown territory), crop a larger, concentric box sized
#             # exactly for this sample's sampled rotation (_rotation_source_shape), rotate
#             # that, and center-crop back down to CROP_SHAPE afterwards. Every voxel of the
#             # final CROP_SHAPE crop then comes from real anatomy, not the background fill.
#             outer_shape = _rotation_source_shape(CROP_SHAPE, rotation, volume_size=256)
#             outer_start, offset, clamped = _expand_crop_bounds(start, CROP_SHAPE, outer_shape, volume_size=256)
#             if clamped:
#                 # `start` was close enough to the volume edge that the outer box couldn't
#                 # stay centered -- see _expand_crop_bounds()'s docstring. Zero information
#                 # loss isn't guaranteed for this sample; surface it instead of silently
#                 # letting a thin sliver of ROTATION_BACKGROUND back into the final crop.
#                 warnings.warn(
#                     f"Rotation outer-crop clamped near the volume boundary for patient "
#                     f"'{self.patients[idx]}' (start={start.tolist()}, outer_shape={outer_shape}); "
#                     f"the final crop may not be fully free of the -1 rotation background fill."
#                 )
#             oe = outer_start + np.array(outer_shape)
#             mask_image = torch.as_tensor(
#                 mask_padded[outer_start[0]:oe[0], outer_start[1]:oe[1], outer_start[2]:oe[2]],
#                 dtype=torch.float32,
#             )
#             t1n_image = t1n_padded[outer_start[0]:oe[0], outer_start[1]:oe[1], outer_start[2]:oe[2]]
#             crop_origin = outer_start
#         else:
#             mask_image = torch.as_tensor(cropped_mask, dtype=torch.float32)
#             t1n_image = t1n_padded[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
#             crop_origin = start

#         # Global coordinate volume: voxel (i, j, k) of the box we just cropped holds its own
#         # (x, y, z) voxel index in the padded 256^3 volume (crop_origin + local index),
#         # *before* any rotation. Built once per sample so that if we rotate below, we rotate
#         # these exact values along with the anatomy instead of recomputing them from the
#         # rotated crop geometry -- i.e. each anatomy voxel keeps the atlas coordinate it
#         # actually has, rather than being relabeled as if it were still axis-aligned.
#         crop_shape_now = tuple(mask_image.shape)  # CROP_SHAPE, or the larger outer_shape
#         local_idx = torch.stack(
#             torch.meshgrid(
#                 torch.arange(crop_shape_now[0], dtype=torch.float32),
#                 torch.arange(crop_shape_now[1], dtype=torch.float32),
#                 torch.arange(crop_shape_now[2], dtype=torch.float32),
#                 indexing="ij",
#             ),
#             dim=0,
#         )  # (3, Dx, Dy, Dz)
#         origin_t = torch.as_tensor(crop_origin, dtype=torch.float32).view(3, 1, 1, 1)
#         coords = local_idx + origin_t  # (3, Dx, Dy, Dz) raw global voxel coordinates

#         if will_rotate:
#             # t1n_image (GT intensity) and coords are both continuous-valued, so they can
#             # share one bilinear resample. mask_image is rotated separately with
#             # nearest-neighbor so it stays exactly {-1, 1} (bilinear would blur the mask
#             # boundary into fractional "half-masked" voxels).
#             continuous = torch.cat([t1n_image.unsqueeze(0), coords], dim=0)  # (4, D, H, W)
#             continuous = _rotate_volume(
#                 continuous, theta, mode="bilinear", background=ROTATION_BACKGROUND
#             )
#             t1n_image, coords = continuous[0], continuous[1:]

#             mask_image = _rotate_volume(
#                 mask_image.unsqueeze(0), theta, mode="nearest", background=ROTATION_BACKGROUND
#             ).squeeze(0)
#             # Nearest-neighbor keeps mask_image at exactly {-1, 1} already, but guard
#             # against float round-off from the shift-by-background trick in _rotate_volume.
#             mask_image = torch.where(mask_image >= 0, 1.0, -1.0)

#             # Center-crop the rotated outer box back down to CROP_SHAPE, using the exact
#             # `offset` from _expand_crop_bounds() -- this reconstructs precisely the
#             # start:start+CROP_SHAPE window crop_mask_to_center() selected (same window used
#             # when will_rotate is False), except every voxel in it now comes from real
#             # rotated anatomy instead of a rotation-background-filled corner.
#             ox, oy, oz = offset
#             cx, cy, cz = CROP_SHAPE
#             t1n_image = t1n_image[ox:ox + cx, oy:oy + cy, oz:oz + cz]
#             mask_image = mask_image[ox:ox + cx, oy:oy + cy, oz:oz + cz]
#             coords = coords[:, ox:ox + cx, oy:oy + cy, oz:oz + cz]

#         # voided_image is derived *after* rotation (and after cropping back down to
#         # CROP_SHAPE) so the "hole" always matches the (possibly rotated) mask exactly --
#         # rotating voided_image and mask_image independently could leave their edges
#         # slightly misaligned.
#         voided_image = t1n_image.clone()
#         voided_image[mask_image == 1] = -1

#         t1n_image = t1n_image.unsqueeze(0)
#         mask_image = mask_image.unsqueeze(0)
#         voided_image = voided_image.unsqueeze(0)

#         # Stack the images along the first dimension
#         stacked_image = torch.cat((voided_image, mask_image, t1n_image), dim=0)

#         image = stacked_image[:2,...]
#         label = stacked_image[2, ...].unsqueeze(0)
#         if not self._shape_printed:
#             print("training image:", image.shape)
#             print("training label:", label.shape)
#             self._shape_printed = True
#         # (3, Dx, Dy, Dz) raw global coordinate volume -- possibly rotated above. Consumed
#         # by GaussianDiffusion.training_losses() -> downsample_and_normalize_coords() to
#         # build the 3 extra (x, y, z) coordinate channels; training_losses reuses these
#         # values as-is rather than rebuilding a coordinate grid from the crop geometry, so
#         # rotation augmentation keeps every voxel tagged with its true atlas location.
#         return image, label, coords

#     def crop_mask_to_center(self, mask):
#         # Check if the input mask is of the correct shape
#         if mask.shape != (256, 256, 256):
#             raise ValueError("Input mask must be of shape (256, 256, 256)")

#         # Find the indices of the mask where the value is 1
#         indices = np.argwhere(mask == 1)

#         if indices.size == 0:
#             raise ValueError("The mask does not contain any 1's.")











        


#         #Centroid＋bounding-box constraint
#         centroid = np.mean(indices, axis=0)
#         bbox_min = indices.min(axis=0)
#         bbox_max = indices.max(axis=0)
#         start = _crop_start_centroid_bbox(centroid, bbox_min, bbox_max, crop_size=CROP_SHAPE)
#         end = start + np.array(CROP_SHAPE)






#         # Crop the mask to the center
#         cropped_mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

#         return cropped_mask, start, end

#     def load_data(self, load_dir):
#         img = nib.load(load_dir)
#         data = img.get_fdata()

#         data = data.astype(np.float32)

#         if data.ndim == 2:
#             data = np.expand_dims(data, axis=0)
#             data = np.expand_dims(data, axis=1)

#         return np.asarray(data, dtype=np.float32)



# class BraTSVolumesTest(torch.utils.data.Dataset):
#     def __init__(self, folder1):
#         # Collect all patient folders
#         self.patients = sorted([d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
#         self.folder1 = folder1

#         self.labels = [0] * len(self.patients)  # Assuming all images have the same label for now
#         self._shape_printed = False
#     def __len__(self):
#         return len(self.patients)

#     def __getitem__(self, idx):
#         patient_folder = os.path.join(self.folder1, self.patients[idx])

#         mask_name = self.patients[idx] + '-mask.nii.gz'
#         # mask_name = self.patients[idx] + '-mask-healthy.nii.gz'
#         t1n_name = self.patients[idx] + '-t1n-voided.nii.gz'
#         mask_file = os.path.join(patient_folder, mask_name)
#         t1n_file = os.path.join(patient_folder, t1n_name)

#         mask_image = self.load_data(mask_file)
#         mask_image[mask_image != 1] = -1
#         mask_image = np.pad(mask_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
#         cropped_masks, starts, ends, labeled_masks = self.crop_mask_to_center(mask_image)

#         t1n_image = torch.as_tensor(self.load_data(t1n_file), dtype=torch.float32)
#         t1n_image[t1n_image < 0] = 0
#         max_v = torch.max(t1n_image)
#         if max_v.item() <= 0:
#             raise ValueError(
#                 f"Maximum intensity must be positive for {t1n_file}, got {max_v.item()}"
#             )
#         t1n_image = 2.0 * (t1n_image / max_v) - 1.0
#         t1n_image = torch.as_tensor(
#             np.pad(
#                 t1n_image.cpu().numpy(),
#                 ((8, 8), (8, 8), (50, 51)),
#                 mode="constant",
#                 constant_values=-1,
#             ),
#             dtype=torch.float32,
#         )

#         # Process each cropped mask
#         stacked_images = []
#         for cropped_mask, start, end, labeled_mask in zip(cropped_masks, starts, ends, labeled_masks):
#             cropped_mask_tensor = torch.as_tensor(cropped_mask, dtype=torch.float32)

#             # Crop the T1n image based on the mask
#             voided_image = t1n_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
#             voided_image_full = t1n_image.clone()

#             stacked_image = torch.cat((voided_image.unsqueeze(0), cropped_mask_tensor.unsqueeze(0)), dim=0)
#             stacked_images.append(stacked_image)

#         # `starts` (one crop origin per connected component, in the padded 256^3 frame) is
#         # already threaded out here; generation_val.py/generation_sample.py pass starts[i]
#         # into diffusion.p_sample_loop(..., crop_start=...) to build coordinate channels
#         # matching what training_losses() saw for the same crop location.
#         return stacked_images, voided_image_full, starts, ends, labeled_masks, t1n_name

#     def crop_mask_to_center(self, mask):
#         if mask.shape != (256, 256, 256):
#             raise ValueError("Input mask must be of shape (256, 256, 256)")

#         # Find connected components in the mask using skimage
#         labeled_mask = label(mask, background=-1, connectivity=3)  # Use 3D connectivity
#         num_features = labeled_mask.max()  # The maximum label value corresponds to the number of features

#         if num_features == 0:  # No components found
#             raise ValueError("The mask does not contain any 1's.")

#         cropped_masks = []
#         starts = []
#         ends = []
#         labeled_masks = []

#         for i in range(1, num_features + 1):  # Iterate over each component (skip the background label 0)
#             # Get the indices of the current component
#             indices = np.argwhere(labeled_mask == i)
#             labeled_mask_copy = labeled_mask.copy()
#             labeled_mask_copy[labeled_mask_copy!=i] = -1

#             #Centroid＋bounding-box constraint

#             centroid = np.mean(indices, axis=0)
#             bbox_min = indices.min(axis=0)
#             bbox_max = indices.max(axis=0)
#             start = _crop_start_centroid_bbox(centroid, bbox_min, bbox_max, crop_size=CROP_SHAPE)
#             end = start + np.array(CROP_SHAPE)
















#             # Crop the mask to the center
#             cropped_mask = labeled_mask_copy[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
#             cropped_mask[cropped_mask!=-1] = 1
#             cropped_masks.append(cropped_mask)
#             starts.append(start)
#             ends.append(end)
#             labeled_masks.append(labeled_mask_copy)

#         return cropped_masks, starts, ends, labeled_masks

#     def load_data(self, load_dir):
#             img = nib.load(load_dir)
#             data = img.get_fdata()

#             data = data.astype(np.float32)

#             if data.ndim == 2:
#                 data = np.expand_dims(data, axis=0)
#                 data = np.expand_dims(data, axis=1)

#             return torch.as_tensor(data, dtype=torch.float32)
    

# class BraTSVolumesVal(torch.utils.data.Dataset):
#     def __init__(self, folder1):
#         # Collect all patient folders
#         self.patients = sorted([d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
#         self.folder1 = folder1

#         self.labels = [0] * len(self.patients)  # Assuming all images have the same label for now

#     def __len__(self):
#         return len(self.patients)

#     def __getitem__(self, idx):
#         patient_folder = os.path.join(self.folder1, self.patients[idx])

#         # mask_name = self.patients[idx] + '-mask.nii.gz'
#         mask_name = self.patients[idx] + '-mask-healthy.nii.gz'
#         t1n_name = self.patients[idx] + '-t1n-voided.nii.gz'
#         mask_file = os.path.join(patient_folder, mask_name)
#         t1n_file = os.path.join(patient_folder, t1n_name)

#         mask_image = self.load_data(mask_file)
#         mask_image[mask_image != 1] = -1
#         mask_image = np.pad(mask_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
#         cropped_masks, starts, ends, labeled_masks = self.crop_mask_to_center(mask_image)

#         t1n_image = torch.as_tensor(self.load_data(t1n_file), dtype=torch.float32)
#         t1n_image[t1n_image < 0] = 0
#         max_v = torch.max(t1n_image)
#         if max_v.item() <= 0:
#             raise ValueError(
#                 f"Maximum intensity must be positive for {t1n_file}, got {max_v.item()}"
#             )
#         t1n_image = 2.0 * (t1n_image / max_v) - 1.0
#         t1n_image = torch.as_tensor(
#             np.pad(
#                 t1n_image.cpu().numpy(),
#                 ((8, 8), (8, 8), (50, 51)),
#                 mode="constant",
#                 constant_values=-1,
#             ),
#             dtype=torch.float32,
#         )

#         # Process each cropped mask
#         stacked_images = []
#         for cropped_mask, start, end, labeled_mask in zip(cropped_masks, starts, ends, labeled_masks):
#             cropped_mask_tensor = torch.as_tensor(cropped_mask, dtype=torch.float32)

#             # Crop the T1n image based on the mask
#             voided_image = t1n_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
#             voided_image_full = t1n_image.clone()

#             stacked_image = torch.cat((voided_image.unsqueeze(0), cropped_mask_tensor.unsqueeze(0)), dim=0)
#             stacked_images.append(stacked_image)

#         # Same `starts` propagation as BraTSVolumesTest.__getitem__ above; needed by the
#         # sampling loop to build coordinate channels for this crop.
#         return stacked_images, voided_image_full, starts, ends, labeled_masks, t1n_name

#     def crop_mask_to_center(self, mask):
#         if mask.shape != (256, 256, 256):
#             raise ValueError("Input mask must be of shape (256, 256, 256)")

#         # Find connected components in the mask using skimage
#         labeled_mask = label(mask, background=-1, connectivity=3)  # Use 3D connectivity
#         num_features = labeled_mask.max()  # The maximum label value corresponds to the number of features

#         if num_features == 0:  # No components found
#             raise ValueError("The mask does not contain any 1's.")

#         cropped_masks = []
#         starts = []
#         ends = []
#         labeled_masks = []

#         for i in range(1, num_features + 1):  # Iterate over each component (skip the background label 0)
#             # Get the indices of the current component
#             indices = np.argwhere(labeled_mask == i)
#             labeled_mask_copy = labeled_mask.copy()
#             labeled_mask_copy[labeled_mask_copy!=i] = -1

#             #centroid + bounding box constraint
#             centroid = np.mean(indices, axis=0)
#             bbox_min = indices.min(axis=0)
#             bbox_max = indices.max(axis=0)
#             start = _crop_start_centroid_bbox(centroid, bbox_min, bbox_max, crop_size=CROP_SHAPE)
#             end = start + np.array(CROP_SHAPE)


















#             # Crop the mask to the center
#             cropped_mask = labeled_mask_copy[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
#             cropped_mask[cropped_mask!=-1] = 1
#             cropped_masks.append(cropped_mask)
#             starts.append(start)
#             ends.append(end)
#             labeled_masks.append(labeled_mask_copy)

#         return cropped_masks, starts, ends, labeled_masks

#     def load_data(self, load_dir):
#             img = nib.load(load_dir)
#             data = img.get_fdata()

#             data = data.astype(np.float32)

#             if data.ndim == 2:
#                 data = np.expand_dims(data, axis=0)
#                 data = np.expand_dims(data, axis=1)

#             return torch.as_tensor(data, dtype=torch.float32)



import os
import os.path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from scipy.ndimage import distance_transform_edt
from skimage.measure import label


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, folder1, use_augmented=True):
        # Collect all patient folders
        patients = sorted([d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
        self.folder1 = folder1

        # Each sample is (patient, suffix): suffix='' is the original {patient}-mask-healthy.nii.gz
        # / {patient}-t1n.nii.gz pair; suffix='-0001'..'-0009' selects a precomputed augmented
        # mask-healthy variant (different synthetic lesion placement/shape). __getitem__ falls
        # back to the original t1n.nii.gz for indices that don't have a matching
        # {patient}-t1n{suffix}.nii.gz (only indices 0005-0009 also perturb the anatomy itself).
        self.samples = []
        for patient in patients:
            self.samples.append((patient, ''))
            if use_augmented:
                patient_folder = os.path.join(folder1, patient)
                for i in range(1, 10):
                    suffix = f'-{i:04d}'
                    if os.path.exists(os.path.join(patient_folder, patient + '-mask-healthy' + suffix + '.nii.gz')):
                        self.samples.append((patient, suffix))

        self.labels = [0] * len(self.samples)  # Assuming all images have the same label for now

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient, suffix = self.samples[idx]
        patient_folder = os.path.join(self.folder1, patient)

        mask_name = patient + '-mask-healthy' + suffix + '.nii.gz'
        t1n_name = patient + '-t1n' + suffix + '.nii.gz'
        mask_file = os.path.join(patient_folder, mask_name)
        t1n_file = os.path.join(patient_folder, t1n_name)
        if not os.path.exists(t1n_file):
            # This augmentation index only varies the mask placement; anatomy is unchanged.
            t1n_file = os.path.join(patient_folder, patient + '-t1n.nii.gz')

        # Load mask
        mask_image = self.load_data(mask_file)
        #nib.save(nib.Nifti1Image(np.asarray(mask_image), None), 'mask_loaded.nii.gz')
        # Set non-one values to minus 1
        mask_image[mask_image != 1] = -1
        #nib.save(nib.Nifti1Image(np.asarray(mask_image), None), 'mask_set_to_minus1.nii.gz')
        # Pad mask
        mask_image = np.pad(mask_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
        #nib.save(nib.Nifti1Image(np.asarray(mask_image), None), 'mask_after_pad.nii.gz')
        full_padded_mask = mask_image  # kept for the SDT cache miss path below, since
        # crop_mask_to_center() reassigns mask_image to the cropped array next

        # Crop to desired region
        mask_image, start, end = self.crop_mask_to_center(mask_image)
        #nib.save(nib.Nifti1Image(np.asarray(mask_image), None), 'mask_after_crop.nii.gz')
        if mask_image.shape != (128, 128, 128):
            raise ValueError("Generated mask {} must be of shape (128, 128, 128) but is {}!".format(mask_file, mask_image.shape))
        mask_image = torch.as_tensor(mask_image, dtype=torch.float32)

        # Signed distance transform (SDT) of the mask, cropped to this patient's
        # 128^3 window. Cached to disk (~8MB/patient) because crop_mask_to_center()
        # is deterministic here (no rotation augmentation in this class), so a given
        # patient always yields the exact same SDT crop -- recomputing it via
        # distance_transform_edt on every __getitem__ call (every sample, every
        # epoch, for the whole training run) would be pure waste. Cached at 128^3
        # (not the full padded 256^3) since that's the only region ever used, and
        # it's what actually fits on disk here.
        sdt_cache_file = os.path.join(patient_folder, patient + '-mask-sdt' + suffix + '.npy')
        if os.path.exists(sdt_cache_file):
            sdt = np.load(sdt_cache_file)
        else:
            # Computed on the FULL padded 256^3 mask, not just the 128^3 crop, so
            # distances reflect the true lesion geometry even for lesions that
            # extend beyond the crop window; sliced down to the crop afterwards
            # with the same start/end used for mask_image, then that crop (not the
            # full volume) is what gets cached. Positive inside the mask (distance
            # to nearest known voxel), negative outside (distance to nearest
            # missing voxel).
            full_binary_mask = full_padded_mask == 1
            dist_inside = distance_transform_edt(full_binary_mask)
            dist_outside = distance_transform_edt(~full_binary_mask)
            sdt_full = (dist_inside - dist_outside).astype(np.float32)
            sdt = sdt_full[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

            # Write via a temp file (name already ends in .npy so np.save doesn't
            # silently append another one) + atomic rename, so concurrent DataLoader
            # workers racing on the same first-time patient can't leave a
            # corrupt/partial cache file behind.
            tmp_file = os.path.join(
                patient_folder,
                f"{patient}-mask-sdt{suffix}.tmp{os.getpid()}.npy",
            )
            np.save(tmp_file, sdt)
            os.replace(tmp_file, sdt_cache_file)

        sdt = torch.as_tensor(sdt, dtype=torch.float32).unsqueeze(0)  # (1, 128, 128, 128)

        t1n_image = self.load_data(t1n_file)
        t1n_image = np.clip(
            t1n_image,
            np.quantile(t1n_image, 0.005),
            np.quantile(t1n_image, 0.995),
        )
        t1n_image = 2 * (t1n_image - np.min(t1n_image)) / (np.max(t1n_image) - np.min(t1n_image)) - 1
        t1n_image = np.pad(t1n_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
        t1n_image = torch.as_tensor(t1n_image, dtype=torch.float32)
        #t1n_image = self.normalize(t1n_image)
        t1n_image = t1n_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        mask_idx = mask_image.clone()

        voided_image = t1n_image.clone()
        voided_image[mask_image == 1] = -1

        t1n_image = t1n_image.unsqueeze(0)
        mask_image = mask_image.unsqueeze(0)
        voided_image = voided_image.unsqueeze(0)

        # Stack the images along the first dimension
        stacked_image = torch.cat((voided_image, mask_image, t1n_image), dim=0)

        image = stacked_image[:2,...]
        label = stacked_image[2, ...].unsqueeze(0)
        # Global voxel-coordinate volume for the current 128³ crop.
# `start` is the crop origin inside the padded 256³ volume.
        crop_d, crop_h, crop_w = t1n_image.shape[-3:]

        local_coords = torch.stack(
            torch.meshgrid(
                torch.arange(crop_d, dtype=torch.float32),
                torch.arange(crop_h, dtype=torch.float32),
                torch.arange(crop_w, dtype=torch.float32),
                indexing="ij",
            ),
            dim=0,
        )  # [3, 128, 128, 128]

        crop_origin = torch.as_tensor(
            start,
            dtype=torch.float32,
        ).view(3, 1, 1, 1)

        coords = local_coords + crop_origin

        # return image, label
        return image, label, coords, sdt

    def crop_mask_to_center(self, mask):
        # Check if the input mask is of the correct shape
        if mask.shape != (256, 256, 256):
            raise ValueError("Input mask must be of shape (256, 256, 256)")

        # Find the indices of the mask where the value is 1
        indices = np.argwhere(mask == 1)

        if indices.size == 0:
            raise ValueError("The mask does not contain any 1's.")

        # Calculate the centroid of the mask
        centroid = np.mean(indices, axis=0).astype(int)

        # Define the cropping indices
        start = centroid - 64  # 128 / 2 = 64
        end = centroid + 64  # 128 / 2 = 64
        # Ensure the cropping indices are within bounds and maintain a size of 128
        for i in range(3):
            if start[i] < 0:
                start[i] = 0
                end[i] = 128
            elif end[i] > 256:
                end[i] = 256
                start[i] = 256 - 128
        # Ensure the difference between start and end is always 128
        for i in range(3):
            if end[i] - start[i] != 128:
                if end[i] - start[i] < 128:
                    # If the range is less than 128, adjust the start
                    start[i] = end[i] - 128
                else:
                    # If the range is more than 128, adjust the end
                    end[i] = start[i] + 128
        # Crop the mask to the center
        cropped_mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        return cropped_mask, start, end

    def load_data(self, load_dir):
        img = nib.load(load_dir)
        data = img.get_fdata()

        data = data.astype(np.float32)

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=1)

        return np.asarray(data, dtype=np.float32)



class BraTSVolumesTest(torch.utils.data.Dataset):
    def __init__(self, folder1):
        # Collect all patient folders
        self.patients = sorted([d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))])
        self.folder1 = folder1

        self.labels = [0] * len(self.patients)  # Assuming all images have the same label for now

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_folder = os.path.join(self.folder1, self.patients[idx])

        mask_name = self.patients[idx] + '-mask.nii.gz'
        # mask_name = self.patients[idx] + '-mask-healthy.nii.gz'
        t1n_name = self.patients[idx] + '-t1n-voided.nii.gz'
        mask_file = os.path.join(patient_folder, mask_name)
        t1n_file = os.path.join(patient_folder, t1n_name)

        mask_image = self.load_data(mask_file)
        mask_image[mask_image != 1] = -1
        mask_image = np.pad(mask_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1)
        full_mask = torch.as_tensor(mask_image.copy(), dtype=torch.float32)
        cropped_masks, starts, ends, labeled_masks = self.crop_mask_to_center(mask_image)

        t1n_image = self.load_data(t1n_file)
        t1n_image = torch.as_tensor(t1n_image, dtype=torch.float32)
        t1n_image = np.clip(
            t1n_image,
            np.quantile(t1n_image, 0.005),
            np.quantile(t1n_image, 0.995),
        )
        t1n_image = 2 * (t1n_image - torch.min(t1n_image)) / (torch.max(t1n_image) - torch.min(t1n_image)) - 1
        t1n_image = torch.tensor(np.pad(t1n_image, ((8, 8), (8, 8), (50, 51)), 'constant', constant_values=-1))
       
        # Process each cropped mask
        stacked_images = []
        for cropped_mask, start, end, labeled_mask in zip(cropped_masks, starts, ends, labeled_masks):
            cropped_mask_tensor = torch.as_tensor(cropped_mask, dtype=torch.float32)

            # Crop the T1n image based on the mask
            voided_image = t1n_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            voided_image_full = t1n_image.clone()

            stacked_image = torch.cat((voided_image.unsqueeze(0), cropped_mask_tensor.unsqueeze(0)), dim=0)
            stacked_images.append(stacked_image)
        # return (
        #     stacked_images,
        #     voided_image_full,
        #     starts,
        #     ends,
        #     labeled_masks,
        #     t1n_name,
        # )
        return (
            stacked_images,
            voided_image_full,
            full_mask,
            starts,
            ends,
            labeled_masks,
            t1n_name,
        )
        # return stacked_images, voided_image_full, starts, ends, labeled_masks, t1n_name

    def crop_mask_to_center(self, mask):
        if mask.shape != (256, 256, 256):
            raise ValueError("Input mask must be of shape (256, 256, 256)")

        # Find connected components in the mask using skimage
        labeled_mask = label(mask, background=-1, connectivity=3)  # Use 3D connectivity
        num_features = labeled_mask.max()  # The maximum label value corresponds to the number of features

        if num_features == 0:  # No components found
            raise ValueError("The mask does not contain any 1's.")

        cropped_masks = []
        starts = []
        ends = []
        labeled_masks = []

        for i in range(1, num_features + 1):  # Iterate over each component (skip the background label 0)
            # Get the indices of the current component
            indices = np.argwhere(labeled_mask == i)
            labeled_mask_copy = labeled_mask.copy()
            labeled_mask_copy[labeled_mask_copy!=i] = -1

            # Calculate the centroid of the component
            centroid = np.mean(indices, axis=0).astype(int)

            # Define the cropping indices
            start = centroid - 64  # 128 / 2 = 64
            end = centroid + 64  # 128 / 2 = 64
            # Ensure the cropping indices are within bounds and maintain a size of 128
            for j in range(3):
                if start[j] < 0:
                    start[j] = 0
                    end[j] = 128
                elif end[j] > 256:
                    end[j] = 256
                    start[j] = 256 - 128
            # Ensure the difference between start and end is always 128
            for j in range(3):
                if end[j] - start[j] != 128:
                    if end[j] - start[j] < 128:
                        # If the range is less than 128, adjust the start
                        start[j] = end[j] - 128
                    else:
                        # If the range is more than 128, adjust the end
                        end[j] = start[j] + 128
            # Crop the mask to the center
            cropped_mask = labeled_mask_copy[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            cropped_mask[cropped_mask!=-1] = 1
            cropped_masks.append(cropped_mask)
            starts.append(start)
            ends.append(end)
            labeled_masks.append(labeled_mask_copy)

        return cropped_masks, starts, ends, labeled_masks

    def load_data(self, load_dir):
            img = nib.load(load_dir)
            data = img.get_fdata()

            data = data.astype(np.float32)

            if data.ndim == 2:
                data = np.expand_dims(data, axis=0)
                data = np.expand_dims(data, axis=1)

            return torch.as_tensor(data, dtype=torch.float32)