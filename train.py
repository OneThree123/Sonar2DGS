import os
import sys
import uuid
import atexit
import math
import random
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import (
    network_gui,
    normalize_render_intensity_for_output,
    prepare_sonar_intensity_for_loss,
    render,
)
from scene import GaussianModel, Scene
from scene.dataset_readers import readBasicInfo, readSonarCameras
from utils.general_utils import PILtoTorch, safe_state
from utils.group_schedule import GroupTrainingSchedule
from utils.image_utils import psnr, render_net_image
from utils.intensity_diagnostics import format_raw_intensity_stats
from utils.loss_curve_logger import LossCurveLogger
from utils.loss_utils import gaussian_blur, l1_loss, masked_l1_loss, masked_ssim, weighted_mean

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


OPACITY_RESET_MODE_ALIASES = {
    "interval": "interval",
    "every": "interval",
    "repeat": "interval",
    "repeated": "interval",
    "once": "once",
    "single": "once",
    "none": "none",
    "never": "none",
    "off": "none",
    "disable": "none",
    "disabled": "none",
}


def normalize_opacity_reset_mode(mode):
    normalized = str(mode or "interval").lower()
    if normalized not in OPACITY_RESET_MODE_ALIASES:
        valid_modes = ", ".join(sorted(set(OPACITY_RESET_MODE_ALIASES.values())))
        raise ValueError(
            f"Unknown opacity reset mode: {mode!r}. Expected one of: {valid_modes}."
        )
    return OPACITY_RESET_MODE_ALIASES[normalized]


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)

    print("Tensorboard not available: not logging progress")
    return None


def get_total_gaussians(gaussians):
    return int(gaussians.get_xyz.shape[0])


def annealed_lowpass_variance(
    iteration,
    initial_variance,
    final_variance,
    from_iteration,
    until_iteration,
):
    initial_sigma = math.sqrt(float(initial_variance))
    final_sigma = math.sqrt(float(final_variance))
    if iteration <= from_iteration:
        return initial_sigma * initial_sigma
    if iteration >= until_iteration:
        return final_sigma * final_sigma

    progress = (iteration - from_iteration) / float(until_iteration - from_iteration)
    sigma = initial_sigma + (final_sigma - initial_sigma) * progress
    return sigma * sigma


def scheduled_max_gaussians(
    iteration,
    initial_limit,
    anneal_target_limit,
    post_anneal_limit,
    until_iteration,
):
    initial_limit = int(initial_limit)
    anneal_target_limit = int(anneal_target_limit)
    post_anneal_limit = int(post_anneal_limit)
    if iteration > until_iteration:
        return post_anneal_limit
    if iteration <= 0:
        return initial_limit
    if iteration == until_iteration:
        return anneal_target_limit

    progress = iteration / float(until_iteration)
    ratio = anneal_target_limit / float(initial_limit)
    return min(
        anneal_target_limit,
        max(initial_limit, int(round(initial_limit * ratio**progress))),
    )


def print_group_event_schedule(name, events, schedule):
    if not events:
        print(f"[Group schedule] {name}: disabled")
        return

    for group, requested_iterations in events.items():
        requested = ", ".join(str(value) for value in requested_iterations)
        merged = " (merged)" if len(requested_iterations) > 1 else ""
        print(
            f"[Group schedule] {name}: requested iteration(s) {requested} "
            f"-> group {group} -> effective iteration "
            f"{schedule.group_end_iteration(group)}{merged}"
        )


def adjust_progress_bar_elapsed(progress_bar, seconds):
    if seconds <= 0:
        return

    for attr in ("start_t", "last_print_t"):
        if hasattr(progress_bar, attr):
            setattr(progress_bar, attr, getattr(progress_bar, attr) + seconds)


def filter_small_connected_components(mask, min_component_size):
    if min_component_size <= 1 or not mask.any():
        return mask

    try:
        from scipy import ndimage

        labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if component_count == 0:
            return mask
        component_sizes = np.bincount(labels.ravel())
        keep_labels = component_sizes >= int(min_component_size)
        keep_labels[0] = False
        return keep_labels[labels]
    except Exception:
        return filter_small_connected_components_fallback(mask, min_component_size)


def filter_small_connected_components_fallback(mask, min_component_size):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue

            stack = [(y, x)]
            component = []
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not visited[ny, nx]
                        and mask[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(component) >= min_component_size:
                ys, xs = zip(*component)
                keep[ys, xs] = True

    return keep


def filter_by_neighbor_count(mask, min_neighbors):
    if min_neighbors <= 1 or not mask.any():
        return mask

    mask_float = mask.to(dtype=torch.float32)[None, None]
    kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=torch.float32)
    neighbor_count = F.conv2d(mask_float, kernel, padding=1)[0, 0]
    return mask & (neighbor_count >= int(min_neighbors))


def binary_opening(mask, iterations):
    if iterations <= 0 or not mask.any():
        return mask

    mask_float = mask.to(dtype=torch.float32)[None, None]
    kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=torch.float32)
    for _ in range(int(iterations)):
        eroded = F.conv2d(mask_float, kernel, padding=1) >= 9.0
        dilated = F.conv2d(eroded.to(dtype=torch.float32), kernel, padding=1) > 0.0
        mask_float = dilated.to(dtype=torch.float32)
    return mask_float[0, 0] > 0.0


def anti_noise_background_rows(height, data_type):
    data_type = str(data_type or "").upper()
    if data_type == "UXO_REAL":
        start = min(233, height - 1)
        effective_height = min(780, height - start)
        sample_height = max(1, int(effective_height * 0.1))
        return start, min(height, start + sample_height)

    sample_height = max(1, int(height * 0.1))
    return 0, min(height, sample_height)


def anti_noise_background_sample(source, data_type):
    data_type = str(data_type or "").upper()
    height = source.shape[-2]
    width = source.shape[-1]

    if data_type == "MONO_REAL":
        start = min(5, height - 1)
        effective_height = height - start
        sample_height = max(1, int(effective_height * 0.1))
        end = min(height, start + sample_height)
        row_sample = source[..., start:end, :]

        if width <= 4:
            return row_sample[..., :0]

        middle_start = width // 2 - 2
        middle_end = width // 2 + 2
        left_sample = row_sample[..., :middle_start]
        right_sample = row_sample[..., middle_end:]
        return torch.cat((left_sample, right_sample), dim=-1)

    bg_start, bg_end = anti_noise_background_rows(height, data_type)
    return source[..., bg_start:bg_end, :]


def anti_noise_clear_top_end(height, clear_top_ratio):
    return min(height, int(height * clear_top_ratio))


@torch.no_grad()
def build_anti_noise_image(
    image,
    min_component_size,
    min_neighbors,
    open_iterations,
    clear_top_ratio,
    data_type="",
    valid_mask=None,
):
    source = image.detach()
    if valid_mask is not None:
        valid_mask = valid_mask.detach().to(device=source.device)
        if valid_mask.shape[-2:] != source.shape[-2:]:
            raise ValueError(
                "Anti-noise mask spatial shape must match the image: "
                f"image={tuple(source.shape)}, mask={tuple(valid_mask.shape)}"
            )
        while valid_mask.ndim < source.ndim:
            valid_mask = valid_mask.unsqueeze(0)
        try:
            valid_mask = valid_mask.expand_as(source)
        except RuntimeError as exc:
            raise ValueError(
                "Anti-noise mask shape is not broadcastable to the image: "
                f"image={tuple(source.shape)}, mask={tuple(valid_mask.shape)}"
            ) from exc
        source = source * (valid_mask >= 0.5).to(dtype=source.dtype)

    background_sample = anti_noise_background_sample(source, data_type)
    if background_sample.numel() == 0:
        raise ValueError(
            "Anti-noise background sample is empty. "
            f"data_type={data_type!r}, image shape={tuple(source.shape)}"
        )
    background_mean = background_sample.mean()
    height = source.shape[-2]

    denoised = torch.clamp(source - background_mean, min=0.0)
    nonzero_mask = denoised > 0
    if denoised.ndim > 2:
        nonzero_mask = nonzero_mask.reshape(-1, denoised.shape[-2], denoised.shape[-1]).any(dim=0)

    filtered_mask = filter_by_neighbor_count(nonzero_mask, min_neighbors)
    filtered_mask = binary_opening(filtered_mask, open_iterations)
    clear_top_height = anti_noise_clear_top_end(height, clear_top_ratio)
    if clear_top_height > 0:
        filtered_mask[:clear_top_height, :] = False

    component_mask = filter_small_connected_components(
        filtered_mask.cpu().numpy().astype(bool),
        int(min_component_size),
    )
    component_mask = torch.from_numpy(component_mask).to(device=denoised.device, dtype=torch.bool)
    while component_mask.ndim < denoised.ndim:
        component_mask = component_mask.unsqueeze(0)

    denoised = denoised * component_mask.to(dtype=denoised.dtype)
    removed_pixels = int(nonzero_mask.sum().item() - component_mask.reshape(-1, *component_mask.shape[-2:]).any(dim=0).sum().item())
    return denoised, float(background_mean.item()), removed_pixels


@torch.no_grad()
def prepare_anti_noise_ground_truths(
    scene,
    min_component_size,
    min_neighbors,
    open_iterations,
    clear_top_ratio,
):
    cameras = []
    seen_camera_ids = set()
    for camera in list(scene.getTrainCameras()) + list(scene.getTestCameras()):
        camera_id = id(camera)
        if camera_id in seen_camera_ids:
            continue
        seen_camera_ids.add(camera_id)
        cameras.append(camera)

    print(
        "[Anti-noise] Preparing {} reference images "
        "(data-type background mean subtraction, min neighbors {}, open iterations {}, "
        "min component size {}, clear top ratio {}).".format(
            len(cameras),
            int(min_neighbors),
            int(open_iterations),
            int(min_component_size),
            float(clear_top_ratio),
        ),
        flush=True,
    )

    removed_pixels_total = 0
    for camera in cameras:
        anti_noise_image, background_mean, removed_pixels = build_anti_noise_image(
            camera.original_image,
            min_component_size,
            min_neighbors,
            open_iterations,
            clear_top_ratio,
            getattr(camera, "data_type", ""),
            getattr(camera, "gt_alpha_mask", None),
        )
        camera.anti_noise_image = anti_noise_image
        camera.anti_noise_background_mean = background_mean
        removed_pixels_total += removed_pixels

    print(
        "[Anti-noise] Done. Removed {} foreground candidate pixels by isolation/component filtering.".format(
            removed_pixels_total
        ),
        flush=True,
    )


def get_anti_noise_settings(opt):
    min_component_size = int(
        getattr(opt, "anti_noise_min_component_size", 256) or 0
    )
    min_neighbors = int(getattr(opt, "anti_noise_min_neighbors", 6) or 0)
    open_iterations = int(getattr(opt, "anti_noise_open_iterations", 1) or 0)
    clear_top_ratio = float(getattr(opt, "anti_noise_clear_top_ratio", 0.1) or 0.0)

    if min_component_size < 0:
        raise ValueError(
            "anti_noise_min_component_size must be non-negative, "
            f"got {min_component_size}"
        )
    if min_neighbors < 0 or min_neighbors > 9:
        raise ValueError(
            "anti_noise_min_neighbors must be in [0, 9], "
            f"got {min_neighbors}"
        )
    if open_iterations < 0:
        raise ValueError(
            "anti_noise_open_iterations must be non-negative, "
            f"got {open_iterations}"
        )
    if clear_top_ratio < 0.0 or clear_top_ratio > 1.0:
        raise ValueError(
            "anti_noise_clear_top_ratio must be in [0, 1], "
            f"got {clear_top_ratio}"
        )

    return min_component_size, min_neighbors, open_iterations, clear_top_ratio


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, loss_curve_plot_interval=100):
    stop_after_training_seconds = float(
        getattr(opt, "stop_after_training_seconds", 0.0) or 0.0
    )
    if stop_after_training_seconds < 0:
        raise ValueError(
            "stop_after_training_seconds must be non-negative, "
            f"got {stop_after_training_seconds}"
        )

    loss_intensity_mode = str(
        getattr(opt, "loss_intensity_mode", "normalized") or "normalized"
    ).lower()
    if loss_intensity_mode not in ("normalized", "mean_match"):
        raise ValueError(
            "Unknown loss intensity mode: "
            f"{loss_intensity_mode!r}. Expected 'normalized' or 'mean_match'."
        )

    opacity_reset_mode = normalize_opacity_reset_mode(
        getattr(opt, "opacity_reset_mode", "interval")
    )
    opacity_reset_interval = int(opt.opacity_reset_interval)
    if opacity_reset_mode != "none" and opacity_reset_interval <= 0:
        raise ValueError(
            "opacity_reset_interval must be positive when opacity reset is enabled, "
            f"got {opacity_reset_interval}"
        )

    lowpass_s_init = float(opt.lowpass_s_init)
    lowpass_s_final = float(opt.lowpass_s_final)
    lowpass_anneal_from_iter = int(opt.lowpass_anneal_from_iter)
    lowpass_anneal_until_iter = int(opt.lowpass_anneal_until_iter)
    if lowpass_s_init <= 0.0 or lowpass_s_final <= 0.0:
        raise ValueError("lowpass_s_init and lowpass_s_final must be positive")
    if lowpass_s_init < lowpass_s_final:
        raise ValueError("lowpass_s_init must be greater than or equal to lowpass_s_final")
    if lowpass_anneal_until_iter <= lowpass_anneal_from_iter:
        raise ValueError(
            "lowpass_anneal_until_iter must be greater than lowpass_anneal_from_iter"
        )

    regularization_from_iter = int(
        getattr(opt, "regularization_from_iter", 2_000)
    )
    if regularization_from_iter < 0:
        raise ValueError("regularization_from_iter must be non-negative")

    max_gaussians_initial = int(opt.max_gaussians_initial)
    max_gaussians_anneal_target = int(opt.max_gaussians_anneal_target)
    max_gaussians_post_anneal = int(opt.max_gaussians)
    max_gaussians_anneal_until_iter = int(opt.max_gaussians_anneal_until_iter)
    if (
        max_gaussians_initial <= 0
        or max_gaussians_anneal_target < max_gaussians_initial
        or max_gaussians_post_anneal < max_gaussians_anneal_target
    ):
        raise ValueError(
            "Gaussian limits must satisfy 0 < max_gaussians_initial <= "
            "max_gaussians_anneal_target <= max_gaussians"
        )
    if max_gaussians_anneal_until_iter <= 0:
        raise ValueError("max_gaussians_anneal_until_iter must be positive")

    first_iter = 0
    completed_groups = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel()
    gaussians.initial_opacity = dataset.initial_opacity

    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = scene.getTrainCameras()
    if not train_cameras:
        raise ValueError("Training requires at least one training camera.")

    group_schedule = GroupTrainingSchedule(
        group_size=len(train_cameras),
        requested_iterations=int(opt.iterations),
    )
    testing_events = group_schedule.quantize_events(testing_iterations)
    saving_events = group_schedule.quantize_events(saving_iterations)
    checkpoint_events = group_schedule.quantize_events(checkpoint_iterations)
    loss_curve_events = group_schedule.periodic_events(loss_curve_plot_interval)
    densify_events = group_schedule.periodic_events(
        opt.densification_interval,
        from_iteration=opt.densify_from_iter,
        until_iteration=opt.densify_until_iter,
        until_inclusive=False,
    )
    if opacity_reset_mode == "interval":
        opacity_reset_events = group_schedule.periodic_events(
            opacity_reset_interval,
            until_iteration=opt.densify_until_iter,
            until_inclusive=False,
        )
    elif opacity_reset_mode == "once":
        opacity_reset_events = group_schedule.quantize_events(
            [opacity_reset_interval]
            if opacity_reset_interval < int(opt.densify_until_iter)
            else []
        )
    else:
        opacity_reset_events = {}
    opacity_size_threshold_step = (
        group_schedule.group_end_iteration(
            group_schedule.group_for_iteration(opacity_reset_interval)
        )
        if opacity_reset_interval > 0
        else None
    )

    effective_testing_iterations = [
        group_schedule.group_end_iteration(group) for group in testing_events
    ]
    effective_saving_iterations = {
        group_schedule.group_end_iteration(group) for group in saving_events
    }
    effective_checkpoint_iterations = {
        group_schedule.group_end_iteration(group) for group in checkpoint_events
    }

    print(
        "[Group schedule] train cameras={}, requested iterations={}, "
        "groups={}, effective iterations={}, rounding overhead={}".format(
            group_schedule.group_size,
            group_schedule.requested_iterations,
            group_schedule.total_groups,
            group_schedule.effective_iterations,
            group_schedule.rounding_overhead,
        )
    )
    if group_schedule.requested_iterations < group_schedule.group_size:
        print(
            "[Group schedule] Warning: requested iterations are fewer than one "
            "complete group; one full group will be trained."
        )
    print_group_event_schedule("test", testing_events, group_schedule)
    print_group_event_schedule("save", saving_events, group_schedule)
    print_group_event_schedule("checkpoint", checkpoint_events, group_schedule)
    print(
        "[Group schedule] densify: {} effective group event(s); "
        "opacity reset: {} effective group event(s); "
        "loss-curve refresh: {} effective group event(s)".format(
            len(densify_events),
            len(opacity_reset_events),
            len(loss_curve_events),
        )
    )

    if len(scene.getTrainCameras()) > 0:
        network_gui.set_sonar_defaults(scene.getTrainCameras()[0])
    elif len(scene.getTestCameras()) > 0:
        network_gui.set_sonar_defaults(scene.getTestCameras()[0])

    anti_noise_enabled = bool(getattr(opt, "anti_noise", False))
    (
        anti_noise_min_component_size,
        anti_noise_min_neighbors,
        anti_noise_open_iterations,
        anti_noise_clear_top_ratio,
    ) = get_anti_noise_settings(opt)
    if anti_noise_enabled:
        prepare_anti_noise_ground_truths(
            scene,
            anti_noise_min_component_size,
            anti_noise_min_neighbors,
            anti_noise_open_iterations,
            anti_noise_clear_top_ratio,
        )

    gaussians.training_setup(opt)
    if getattr(opt, "densify_debug", False):
        print(
            "[DENSIFY DEBUG enabled] "
            f"from={opt.densify_from_iter}, until={opt.densify_until_iter}, "
            f"interval={opt.densification_interval}, "
            f"grad_threshold={opt.densify_grad_threshold}, "
            f"opacity_cull={opt.opacity_cull}, "
            f"opacity_reset_value={opt.opacity_reset_value}, "
            f"percent_dense={opt.percent_dense}, "
            f"opacity_reset_interval={opt.opacity_reset_interval}, "
            f"opacity_reset_mode={opacity_reset_mode}, "
            f"max_gaussians={max_gaussians_initial}->{max_gaussians_anneal_target} "
            f"until_iter={max_gaussians_anneal_until_iter}, "
            f"then={max_gaussians_post_anneal}, "
            f"lowpass_s={lowpass_s_init}->{lowpass_s_final} "
            f"from_iter={lowpass_anneal_from_iter} "
            f"until_iter={lowpass_anneal_until_iter}, "
            f"direction_min_observations={opt.densify_direction_min_observations}, "
            f"direction_min_consistency={opt.densify_direction_min_consistency}, "
            f"clone_offset_factors=[{opt.clone_offset_min_factor}, {opt.clone_offset_max_factor}], "
            f"scene_extent={scene.cameras_extent}",
            flush=True,
        )

    if checkpoint:
        checkpoint_state = torch.load(checkpoint, weights_only=False)
        if isinstance(checkpoint_state, dict) and checkpoint_state.get("version") == 2:
            model_params = checkpoint_state["gaussians"]
            first_iter = int(checkpoint_state["global_step"])
            completed_groups = int(checkpoint_state["completed_groups"])
            checkpoint_group_size = int(checkpoint_state["group_size"])
            if checkpoint_group_size != group_schedule.group_size:
                raise ValueError(
                    "Checkpoint group size does not match the current dataset: "
                    f"{checkpoint_group_size} != {group_schedule.group_size}"
                )
            expected_iteration = completed_groups * group_schedule.group_size
            if first_iter != expected_iteration:
                raise ValueError(
                    "Checkpoint is not at a complete group boundary: "
                    f"global_step={first_iter}, expected={expected_iteration}"
                )
        else:
            model_params, first_iter = checkpoint_state
            first_iter = int(first_iter)
            if first_iter % group_schedule.group_size != 0:
                raise ValueError(
                    "Legacy checkpoint iteration is not at a complete group "
                    f"boundary: iteration={first_iter}, "
                    f"group_size={group_schedule.group_size}."
                )
            completed_groups = first_iter // group_schedule.group_size
            print(
                "[Group schedule] Loaded a legacy checkpoint at complete group "
                f"{completed_groups}; future checkpoints will use format version 2."
            )
        gaussians.restore(model_params, opt)

    if completed_groups > group_schedule.total_groups:
        raise ValueError(
            f"Checkpoint already completed {completed_groups} groups, but the "
            f"current request contains only {group_schedule.total_groups} groups."
        )

    loss_curve_logger = LossCurveLogger(scene.model_path, first_iter, loss_curve_plot_interval)
    atexit.register(loss_curve_logger.close)
    print(
        "[Loss curves] CSV: {}\n"
        "[Loss curves] HTML: {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
            loss_curve_logger.csv_path,
            loss_curve_logger.combined_html_path,
            loss_curve_logger.image_html_path,
            loss_curve_logger.total_html_path,
            loss_curve_logger.regularization_html_path,
            loss_curve_logger.empty_html_path,
            loss_curve_logger.dist_html_path,
            loss_curve_logger.normal_html_path,
            loss_curve_logger.gaussian_html_path,
            loss_curve_logger.training_time_html_path,
        )
    )

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_empty_for_log = 0.0

    progress_bar = tqdm(
        total=group_schedule.effective_iterations,
        initial=first_iter,
        desc="Training progress",
    )
    output_write_seconds = 0.0
    training_timer_start = time.perf_counter()
    training_time_offset = loss_curve_logger.last_training_time_seconds()
    if stop_after_training_seconds > 0:
        print(
            "[Timed stop] Will stop after the current complete group once training "
            f"time reaches {stop_after_training_seconds:.2f}s.",
            flush=True,
        )

    def exclude_output_write_time(seconds):
        nonlocal output_write_seconds
        output_write_seconds += seconds
        adjust_progress_bar_elapsed(progress_bar, seconds)

    def run_output_write(callback):
        start = time.perf_counter()
        try:
            return callback()
        finally:
            exclude_output_write_time(time.perf_counter() - start)

    def current_training_time_seconds():
        elapsed = time.perf_counter() - training_timer_start - output_write_seconds
        return max(0.0, training_time_offset + elapsed)

    global_step = first_iter
    stop_requested = False
    last_densify_group = max(densify_events, default=0)

    for group_idx in range(completed_groups + 1, group_schedule.total_groups + 1):
        group_start_step = group_schedule.group_start_iteration(group_idx)
        group_end_step = group_schedule.group_end_iteration(group_idx)
        lowpass_variance = annealed_lowpass_variance(
            group_start_step,
            lowpass_s_init,
            lowpass_s_final,
            lowpass_anneal_from_iter,
            lowpass_anneal_until_iter,
        )
        gaussian_limit = scheduled_max_gaussians(
            group_end_step,
            max_gaussians_initial,
            max_gaussians_anneal_target,
            max_gaussians_post_anneal,
            max_gaussians_anneal_until_iter,
        )
        gaussians.update_learning_rate(group_start_step)

        group_cameras = train_cameras.copy()
        random.Random(group_idx).shuffle(group_cameras)
        group_loss_sums = {
            "intensity": 0.0,
            "empty": 0.0,
            "dist": 0.0,
            "normal": 0.0,
            "total": 0.0,
        }
        print(
            f"\n[GROUP {group_idx}/{group_schedule.total_groups}] "
            f"iterations {group_start_step + 1}-{group_end_step}, "
            f"cameras={len(group_cameras)}"
        )

        for sample_idx, viewpoint_cam in enumerate(group_cameras, start=1):
            global_step += 1
            iteration = global_step
            is_group_end = sample_idx == group_schedule.group_size
            iter_start.record()

            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                lowpass_variance=lowpass_variance,
            )
            render_intensity = render_pkg["raw_intensity"]
            screenspace_points = render_pkg["screenspace_points"]
            gs_visibility = render_pkg["gs_visibility"]
            gs_raddi = render_pkg["gs_raddi"]

            gt_source = (
                getattr(viewpoint_cam, "anti_noise_image", viewpoint_cam.original_image)
                if anti_noise_enabled
                else viewpoint_cam.original_image
            )
            gt_image = gt_source.to(render_intensity.device)
            foreground_mask = viewpoint_cam.gt_alpha_mask
            if foreground_mask is None:
                foreground_mask = torch.ones_like(gt_image)
            else:
                foreground_mask = torch.clamp(
                    foreground_mask.to(render_intensity.device), 0.0, 1.0
                )

            loss_intensity, _ = prepare_sonar_intensity_for_loss(
                render_intensity,
                gt_image,
                foreground_mask,
                method=getattr(viewpoint_cam, "normalizing_method", "all"),
                loss_mode=loss_intensity_mode,
            )

            Ll1 = masked_l1_loss(loss_intensity, gt_image, foreground_mask)
            ssim_loss_intensity = loss_intensity
            ssim_gt_image = gt_image
            if anti_noise_enabled:
                ssim_loss_intensity = gaussian_blur(
                    loss_intensity, window_size=3, sigma=1.0
                )
                ssim_gt_image = gaussian_blur(
                    gt_image, window_size=3, sigma=1.0
                )
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0
                - masked_ssim(
                    ssim_loss_intensity,
                    ssim_gt_image,
                    foreground_mask,
                )
            )

            background_weight = (1.0 - foreground_mask).pow(2)
            empty_loss = opt.lambda_empty * weighted_mean(
                loss_intensity.pow(2), background_weight
            )

            lambda_normal = (
                opt.lambda_normal
                if group_start_step >= regularization_from_iter
                else 0.0
            )
            lambda_dist = (
                opt.lambda_dist
                if group_start_step >= regularization_from_iter
                else 0.0
            )

            rend_dist = render_pkg["rend_dist"]
            rend_alpha = render_pkg["rend_alpha"]
            rend_normal = render_pkg["rend_normal"]
            surf_normal = render_pkg["surf_normal"]
            surf_normal_mask = render_pkg["surf_normal_mask"].to(
                dtype=rend_alpha.dtype
            )

            normal_error = rend_alpha.detach() - (
                rend_normal * surf_normal
            ).sum(dim=0, keepdim=True)
            normal_loss = lambda_normal * weighted_mean(
                normal_error, surf_normal_mask
            )
            dist_mask = (rend_alpha.detach() > 0.1).to(dtype=rend_dist.dtype)
            dist_loss = lambda_dist * weighted_mean(rend_dist, dist_mask)

            total_loss = loss + empty_loss + dist_loss + normal_loss
            total_loss.backward()
            iter_end.record()

            with torch.no_grad():
                image_loss_value = loss.item()
                empty_loss_value = empty_loss.item()
                dist_loss_value = dist_loss.item()
                normal_loss_value = normal_loss.item()
                total_loss_value = total_loss.item()

                group_loss_sums["intensity"] += image_loss_value
                group_loss_sums["empty"] += empty_loss_value
                group_loss_sums["dist"] += dist_loss_value
                group_loss_sums["normal"] += normal_loss_value
                group_loss_sums["total"] += total_loss_value

                ema_loss_for_log = 0.4 * image_loss_value + 0.6 * ema_loss_for_log
                ema_empty_for_log = (
                    0.4 * empty_loss_value + 0.6 * ema_empty_for_log
                )
                ema_dist_for_log = 0.4 * dist_loss_value + 0.6 * ema_dist_for_log
                ema_normal_for_log = (
                    0.4 * normal_loss_value + 0.6 * ema_normal_for_log
                )

                if group_idx <= last_densify_group:
                    gaussians.max_radii2D[gs_visibility] = torch.max(
                        gaussians.max_radii2D[gs_visibility],
                        gs_raddi[gs_visibility],
                    )
                    gaussians.add_densification_stats(
                        screenspace_points,
                        gs_visibility,
                        gaussians.get_xyz.grad,
                    )

                gaussians.optimizer.step()
                gaussians.enforce_parameter_bounds()
                gaussians.optimizer.zero_grad(set_to_none=True)

                progress_bar.update(1)
                training_time_seconds = current_training_time_seconds()
                if (
                    not stop_requested
                    and stop_after_training_seconds > 0
                    and training_time_seconds >= stop_after_training_seconds
                ):
                    stop_requested = True
                    print(
                        "\n[ITER {}] Training time reached {:.2f}s; "
                        "finishing group {} before forced output and stop.".format(
                            iteration,
                            training_time_seconds,
                            group_idx,
                        )
                    )

                if is_group_end and group_idx in densify_events:
                    requested = densify_events[group_idx]
                    print(
                        f"\n[GROUP {group_idx}] Densifying at effective iteration "
                        f"{group_end_step}; requested iteration(s) {requested}"
                    )
                    size_threshold = (
                        3200
                        if (
                            opacity_size_threshold_step is not None
                            and group_end_step > opacity_size_threshold_step
                        )
                        else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                        max_gaussians=gaussian_limit,
                        debug=getattr(opt, "densify_debug", False),
                        iteration=group_end_step,
                    )

                if is_group_end and group_idx in opacity_reset_events:
                    requested = opacity_reset_events[group_idx]
                    print(
                        f"\n[GROUP {group_idx}] Resetting opacity at effective "
                        f"iteration {group_end_step}; requested iteration(s) "
                        f"{requested}"
                    )
                    gaussians.reset_opacity(opt.opacity_reset_value)

                total_gaussians = get_total_gaussians(gaussians)
                run_output_write(
                    lambda: loss_curve_logger.log(
                        iteration,
                        image_loss_value,
                        total_loss_value,
                        total_gaussians,
                        empty_loss_value,
                        dist_loss_value,
                        normal_loss_value,
                        training_time_seconds,
                        group=group_idx,
                        sample_in_group=sample_idx,
                        refresh_html=(
                            is_group_end and group_idx in loss_curve_events
                        ),
                    )
                )

                if tb_writer is not None:
                    tb_writer.add_scalar(
                        "train_loss/intensity", image_loss_value, iteration
                    )
                    tb_writer.add_scalar(
                        "train_loss/empty", empty_loss_value, iteration
                    )
                    tb_writer.add_scalar(
                        "train_loss/dist", dist_loss_value, iteration
                    )
                    tb_writer.add_scalar(
                        "train_loss/normal", normal_loss_value, iteration
                    )
                    tb_writer.add_scalar(
                        "train_loss/total", total_loss_value, iteration
                    )
                    tb_writer.add_scalar(
                        "iter_time",
                        iter_start.elapsed_time(iter_end),
                        iteration,
                    )
                    tb_writer.add_scalar(
                        "total_gaussians", total_gaussians, iteration
                    )
                    tb_writer.add_scalar(
                        "total_points", total_gaussians, iteration
                    )

                if iteration % 10 == 0 or is_group_end:
                    progress_bar.set_postfix({
                        "Group": f"{group_idx}/{group_schedule.total_groups}",
                        "Loss": f"{ema_loss_for_log:.5f}",
                        "empty": f"{ema_empty_for_log:.5f}",
                        "distort": f"{ema_dist_for_log:.5f}",
                        "normal": f"{ema_normal_for_log:.5f}",
                        "Gaussians": f"{total_gaussians}",
                        "GS limit": f"{gaussian_limit}",
                        "LP s": f"{lowpass_variance:.3f}",
                    })

                if is_group_end:
                    group_means = {
                        key: value / group_schedule.group_size
                        for key, value in group_loss_sums.items()
                    }
                    print(
                        "\n[GROUP {}] Complete at iteration {}: "
                        "mean intensity={:.6f}, mean total={:.6f}, "
                        "Gaussians={}".format(
                            group_idx,
                            group_end_step,
                            group_means["intensity"],
                            group_means["total"],
                            total_gaussians,
                        )
                    )
                    if tb_writer is not None:
                        for name, value in group_means.items():
                            tb_writer.add_scalar(
                                f"group_loss/{name}",
                                value,
                                group_idx,
                            )
                        tb_writer.add_scalar(
                            "group/effective_iteration",
                            group_end_step,
                            group_idx,
                        )
                        tb_writer.add_scalar(
                            "group/total_gaussians",
                            total_gaussians,
                            group_idx,
                        )

                    training_report(
                        tb_writer,
                        group_end_step,
                        effective_testing_iterations,
                        scene,
                        render,
                        (pipe, 1.0, lowpass_variance),
                        log_intensity_stats=getattr(
                            pipe, "intensity_stats", False
                        ),
                        record_output_write_time=exclude_output_write_time,
                        force=stop_requested,
                        anti_noise_enabled=anti_noise_enabled,
                    )

                    if (
                        group_end_step in effective_saving_iterations
                        or stop_requested
                    ):
                        print(
                            f"\n[GROUP {group_idx} / ITER {group_end_step}] "
                            "Saving Gaussians"
                        )
                        run_output_write(lambda: scene.save(group_end_step))

                    if (
                        group_end_step in effective_checkpoint_iterations
                        or stop_requested
                    ):
                        print(
                            f"\n[GROUP {group_idx} / ITER {group_end_step}] "
                            "Saving Checkpoint"
                        )
                        checkpoint_path = os.path.join(
                            scene.model_path,
                            "chkpnt" + str(group_end_step) + ".pth",
                        )
                        checkpoint_state = {
                            "version": 2,
                            "gaussians": gaussians.capture(),
                            "completed_groups": group_idx,
                            "global_step": group_end_step,
                            "group_size": group_schedule.group_size,
                            "requested_iterations": group_schedule.requested_iterations,
                            "effective_iterations": group_schedule.effective_iterations,
                        }
                        run_output_write(
                            lambda: torch.save(
                                checkpoint_state,
                                checkpoint_path,
                            )
                        )

            with torch.no_grad():
                if network_gui.conn is None:
                    network_gui.try_connect(dataset.render_items)
                while network_gui.conn is not None:
                    try:
                        net_image_bytes = None
                        (
                            custom_cam,
                            do_training,
                            keep_alive,
                            scaling_modifier,
                            render_mode,
                        ) = network_gui.receive()
                        if custom_cam is not None:
                            render_pkg = render(
                                custom_cam,
                                gaussians,
                                pipe,
                                scaling_modifier,
                                lowpass_variance,
                            )
                            render_pkg = normalize_render_intensity_for_output(
                                render_pkg, custom_cam
                            )
                            net_image = render_net_image(
                                render_pkg,
                                dataset.render_items,
                                render_mode,
                                custom_cam,
                            )
                            net_image_bytes = memoryview(
                                (torch.clamp(net_image, min=0, max=1.0) * 255)
                                .byte()
                                .permute(1, 2, 0)
                                .contiguous()
                                .cpu()
                                .numpy()
                            )
                        total_gaussians = get_total_gaussians(gaussians)
                        metrics_dict = {
                            "#": total_gaussians,
                            "loss": ema_loss_for_log,
                            "empty": ema_empty_for_log,
                            "Gaussians": total_gaussians,
                            "Group": group_idx,
                        }
                        network_gui.send(
                            net_image_bytes,
                            dataset.source_path,
                            metrics_dict,
                        )
                        if do_training and (
                            global_step < group_schedule.effective_iterations
                            or not keep_alive
                        ):
                            break
                    except Exception:
                        network_gui.conn = None

        if stop_requested:
            print(
                f"\n[Timed stop] Stopped after completing group {group_idx} "
                f"at iteration {group_end_step}."
            )
            break

    progress_bar.close()

    loss_curve_logger.close()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

def save_tensor_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    image = torch.clamp(image.detach(), 0.0, 1.0).cpu()

    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    elif image.ndim == 3 and image.shape[0] in (3, 4):
        image = image.permute(1, 2, 0)
    elif image.ndim != 2:
        raise ValueError(f"Unsupported image shape for saving: {tuple(image.shape)}")

    array = (image * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(array).save(path)


@torch.no_grad()
def export_test_anti_noise_ground_truths(dataset, opt):
    if not dataset.model_path:
        raise ValueError(
            "--export_anti_noise_ground_truths requires an output folder via -m/--model_path."
        )

    basic_info = readBasicInfo(dataset.source_path)
    test_cameras = readSonarCameras(dataset.source_path, "test", basic_info)
    if not test_cameras:
        raise ValueError(f"No test cameras were found in {dataset.source_path}.")

    min_component_size, min_neighbors, open_iterations, clear_top_ratio = (
        get_anti_noise_settings(opt)
    )
    ground_truth_dir = os.path.join(dataset.model_path, "ground_truth")
    anti_noise_dir = os.path.join(dataset.model_path, "ground_truth_anti_noise")

    print(
        "[Anti-noise export] Writing {} test image pairs to {}.".format(
            len(test_cameras), dataset.model_path
        ),
        flush=True,
    )
    for idx, camera in enumerate(test_cameras):
        resolution = (camera.img_width, camera.img_height)
        ground_truth = PILtoTorch(camera.image, resolution)
        valid_mask = PILtoTorch(camera.mask, resolution) if camera.mask is not None else None
        anti_noise_image, _, _ = build_anti_noise_image(
            ground_truth,
            min_component_size,
            min_neighbors,
            open_iterations,
            clear_top_ratio,
            camera.data_type,
            valid_mask,
        )
        output_name = "{:05d}_{}.png".format(idx, camera.image_name)
        save_tensor_image(ground_truth, os.path.join(ground_truth_dir, output_name))
        save_tensor_image(
            anti_noise_image,
            os.path.join(anti_noise_dir, output_name),
        )

    print(
        "[Anti-noise export] Complete: {} test image pairs written.".format(
            len(test_cameras)
        ),
        flush=True,
    )


@torch.no_grad()
def training_report(
    tb_writer,
    iteration,
    testing_iterations,
    scene,
    render_func,
    render_args,
    log_intensity_stats=False,
    record_output_write_time=None,
    force=False,
    anti_noise_enabled=False,
):
    if not force and iteration not in testing_iterations:
        return

    torch.cuda.empty_cache()
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {"name": "train", "cameras": scene.getTrainCameras()[:5]},
    )

    for config in validation_configs:
        if not config["cameras"]:
            continue

        l1_test = 0.0
        psnr_test = 0.0
        for idx, viewpoint in enumerate(config["cameras"]):
            render_pkg = render_func(viewpoint, scene.gaussians, *render_args)
            if log_intensity_stats:
                print(format_raw_intensity_stats(
                    f"iter={iteration} {config['name']} idx={idx} image={viewpoint.image_name}",
                    render_pkg["raw_intensity"],
                ))
            render_pkg = normalize_render_intensity_for_output(render_pkg, viewpoint)
            image = torch.clamp(render_pkg["r_intensity"], 0.0, 1.0).to("cuda")
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            if config["name"] == "test":
                render_path = os.path.join(
                    scene.model_path,
                    "test_renders",
                    "iteration_{}".format(iteration),
                    "renders",
                    "{:05d}_{}.png".format(idx, viewpoint.image_name),
                )
                save_start = time.perf_counter()
                save_tensor_image(image, render_path)
                if record_output_write_time is not None:
                    record_output_write_time(time.perf_counter() - save_start)

                gt_path = os.path.join(
                    scene.model_path,
                    "test_renders",
                    "iteration_{}".format(iteration),
                    "ground_truth",
                    "{:05d}_{}.png".format(idx, viewpoint.image_name),
                )
                save_start = time.perf_counter()
                save_tensor_image(gt_image, gt_path)
                if record_output_write_time is not None:
                    record_output_write_time(time.perf_counter() - save_start)

                if anti_noise_enabled and hasattr(viewpoint, "anti_noise_image"):
                    anti_noise_gt = viewpoint.anti_noise_image.to("cuda")
                    anti_noise_gt_path = os.path.join(
                        scene.model_path,
                        "test_renders",
                        "iteration_{}".format(iteration),
                        "ground_truth_anti_noise",
                        "{:05d}_{}.png".format(idx, viewpoint.image_name),
                    )
                    save_start = time.perf_counter()
                    save_tensor_image(anti_noise_gt, anti_noise_gt_path)
                    if record_output_write_time is not None:
                        record_output_write_time(time.perf_counter() - save_start)

            if tb_writer and idx < 5:
                tb_writer.add_images(config["name"] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                if testing_iterations and iteration == testing_iterations[0]:
                    tb_writer.add_images(config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(config["cameras"])
        l1_test /= len(config["cameras"])
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))

        if tb_writer:
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint/l1_loss", l1_test, iteration)
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint/psnr", psnr_test, iteration)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 500, 2_000, 7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 500, 2_000, 7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--loss_curve_plot_interval", type=int, default=100)
    parser.add_argument(
        "--export_anti_noise_ground_truths",
        action="store_true",
        help=(
            "Skip training and only export test-set ground_truth and "
            "ground_truth_anti_noise images to the model output folder."
        ),
    )
    args = parser.parse_args(sys.argv[1:])

    if args.export_anti_noise_ground_truths:
        export_test_anti_noise_ground_truths(lp.extract(args), op.extract(args))
        sys.exit(0)

    args.save_iterations.append(args.iterations)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    try:
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.loss_curve_plot_interval,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Loss curves were written to the output folder.")
        sys.exit(130)

    print("\nTraining complete.")
