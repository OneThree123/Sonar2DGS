import torch
import numpy as np
from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    build_scaling_rotation,
    normal_to_rotation,
)
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud


class GaussianModel:
    initial_opacity = 0.1
    scale_min = 0.0
    scale_max = 100.0
    reflection_min = 0.0
    reflection_max = 2.0
    opacity_min = 0.0
    opacity_max = 1.0

    @staticmethod
    def _validate_initial_opacity(initial_opacity):
        initial_opacity = float(initial_opacity)
        if initial_opacity <= 0.0 or initial_opacity > 1.0:
            raise ValueError(
                f"initial_opacity must be in the range (0, 1], got {initial_opacity}"
            )
        return initial_opacity

    def setup_functions(self):
        # def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
        #     RS = build_scaling_rotation(
        #         torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
        #         rotation
        #     ).permute(0, 2, 1)
        #     trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
        #     trans[:, :3, :3] = RS
        #     trans[:, 3, :3] = center
        #     trans[:, 3, 3] = 1
        #     return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        # self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self):
        self._xyz = torch.empty(0)
        self._reflection = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.densify_direction_accum = torch.empty(0)
        self.densify_tangent_grad_accum = torch.empty(0)
        self.densify_direction_denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.densify_direction_min_observations = 4
        self.densify_direction_min_consistency = 0.4
        self.clone_offset_min_factor = 0.2
        self.clone_offset_max_factor = 0.75
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._reflection,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.densify_direction_accum,
            self.densify_tangent_grad_accum,
            self.densify_direction_denom,
        )

    def restore(self, model_args, training_args):
        if len(model_args) not in (10, 13):
            raise ValueError(
                f"Unsupported Gaussian checkpoint layout with {len(model_args)} fields"
            )

        (
            self._xyz,
            self._reflection,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args[:10]

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if len(model_args) == 13:
            self.densify_direction_accum = model_args[10]
            self.densify_tangent_grad_accum = model_args[11]
            self.densify_direction_denom = model_args[12]
        self.optimizer.load_state_dict(opt_dict)
        self.enforce_parameter_bounds()

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_reflection(self):
        return self._reflection.clamp(self.reflection_min, self.reflection_max)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling).clamp(self.scale_min, self.scale_max)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity).clamp(self.opacity_min, self.opacity_max)

    @torch.no_grad()
    def enforce_parameter_bounds(self):
        if self._scaling.numel() > 0:
            self._scaling.clamp_(max=np.log(self.scale_max))

        if self._reflection.numel() > 0:
            self._reflection.clamp_(self.reflection_min, self.reflection_max)

        # Opacity is stored as logits and exposed through sigmoid, so get_opacity
        # is already constrained to [0, 1].

    # def get_covariance(self, scaling_modifier=1):
    #     return self.covariance_activation(
    #         self.get_xyz, self.get_scaling, scaling_modifier, self._rotation
    #     )

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points), dtype=torch.float, device="cuda")
        fused_reflection = torch.tensor(np.asarray(pcd.reflections), dtype=torch.float, device="cuda").reshape(-1, 1)
        fused_normals = torch.tensor(np.asarray(pcd.normals), dtype=torch.float, device="cuda")
        if fused_normals.shape != fused_point_cloud.shape:
            raise ValueError(
                f"Point cloud normals must have shape {tuple(fused_point_cloud.shape)}, "
                f"got {tuple(fused_normals.shape)}."
            )

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            1e-7
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = normal_to_rotation(fused_normals)
        initial_opacity = self._validate_initial_opacity(self.initial_opacity)
        if initial_opacity == 1.0:
            opacities = torch.full(
                (fused_point_cloud.shape[0], 1),
                20.0,
                dtype=torch.float,
                device="cuda",
            )
        else:
            opacities = self.inverse_opacity_activation(
                initial_opacity * torch.ones(
                    (fused_point_cloud.shape[0], 1),
                    dtype=torch.float,
                    device="cuda",
                )
            )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._reflection = nn.Parameter(fused_reflection.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.enforce_parameter_bounds()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.densify_direction_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.densify_tangent_grad_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.densify_direction_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.densify_direction_min_observations = max(
            1,
            int(getattr(training_args, "densify_direction_min_observations", 4)),
        )
        self.densify_direction_min_consistency = float(
            getattr(training_args, "densify_direction_min_consistency", 0.4)
        )
        self.clone_offset_min_factor = float(
            getattr(training_args, "clone_offset_min_factor", 0.2)
        )
        self.clone_offset_max_factor = float(
            getattr(training_args, "clone_offset_max_factor", 0.75)
        )

        if not 0.0 <= self.densify_direction_min_consistency <= 1.0:
            raise ValueError("densify_direction_min_consistency must be in [0, 1]")
        if not 0.0 <= self.clone_offset_min_factor <= self.clone_offset_max_factor:
            raise ValueError(
                "clone offset factors must satisfy 0 <= min <= max"
            )

        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._reflection], "lr": training_args.reflection_lr, "name": "reflection"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz", "reflection", "opacity"]
        for i in range(self._scaling.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = build_rotation(self._rotation).detach().cpu().numpy()[:, :, 2]
        reflection = self._reflection.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate(
            (xyz, normals, reflection, opacities, scale, rotation),
            axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    @torch.no_grad()
    def sample_points_from_gaussians(self, initial_opacity=None):
        if initial_opacity is None:
            initial_opacity = self.initial_opacity
        if initial_opacity <= 0:
            raise ValueError(f"initial_opacity must be positive, got {initial_opacity}")

        device = self.get_xyz.device
        dtype = self.get_xyz.dtype
        num_gaussians = self.get_xyz.shape[0]
        if num_gaussians == 0:
            empty_xyz = torch.empty((0, 3), dtype=dtype, device=device)
            empty_attr = torch.empty((0, 1), dtype=dtype, device=device)
            return empty_xyz, empty_xyz, empty_attr, empty_attr

        opacities = self.get_opacity.squeeze(-1)
        samples_per_gaussian = torch.floor(opacities / initial_opacity + 0.5).long()
        samples_per_gaussian = torch.clamp(samples_per_gaussian, min=0)
        total_samples = int(samples_per_gaussian.sum().item())

        if total_samples == 0:
            empty_xyz = torch.empty((0, 3), dtype=dtype, device=device)
            empty_attr = torch.empty((0, 1), dtype=dtype, device=device)
            return empty_xyz, empty_xyz, empty_attr, empty_attr

        gaussian_ids = torch.repeat_interleave(
            torch.arange(num_gaussians, device=device),
            samples_per_gaussian,
        )

        local_samples = torch.zeros((total_samples, 3), dtype=dtype, device=device)
        local_samples[:, :2] = (
            torch.randn((total_samples, 2), dtype=dtype, device=device)
            * self.get_scaling[gaussian_ids]
        )

        rotations = build_rotation(self._rotation[gaussian_ids])
        sampled_xyz = (
            torch.bmm(rotations, local_samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[gaussian_ids]
        )
        sampled_normals = rotations[:, :, 2]
        sampled_reflections = self.get_reflection[gaussian_ids]
        sampled_opacities = opacities[gaussian_ids].unsqueeze(-1)

        return sampled_xyz, sampled_normals, sampled_reflections, sampled_opacities

    def save_sampled_ply(self, path, initial_opacity=None):
        mkdir_p(os.path.dirname(path))

        sampled_xyz, sampled_normals, sampled_reflections, sampled_opacities = (
            self.sample_points_from_gaussians(initial_opacity)
        )

        xyz = sampled_xyz.detach().cpu().numpy()
        normals = sampled_normals.detach().cpu().numpy()
        reflections = sampled_reflections.detach().cpu().numpy()
        opacities = sampled_opacities.detach().cpu().numpy()

        dtype_full = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("reflection", "f4"),
            ("opacity", "f4"),
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        if xyz.shape[0] > 0:
            elements["x"] = xyz[:, 0]
            elements["y"] = xyz[:, 1]
            elements["z"] = xyz[:, 2]
            elements["nx"] = normals[:, 0]
            elements["ny"] = normals[:, 1]
            elements["nz"] = normals[:, 2]
            elements["reflection"] = reflections[:, 0]
            elements["opacity"] = opacities[:, 0]

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self, reset_value=0.01):
        reset_value = float(reset_value)
        if reset_value <= 0.0 or reset_value >= 1.0:
            raise ValueError(
                f"opacity reset value must be in the range (0, 1), got {reset_value}"
            )
        current_opacity = self.get_opacity
        opacities_new = self.inverse_opacity_activation(
            torch.min(current_opacity, torch.ones_like(current_opacity) * reset_value)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1)

        reflection = np.asarray(plydata.elements[0]["reflection"])[..., np.newaxis]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._reflection = nn.Parameter(torch.tensor(reflection, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.enforce_parameter_bounds()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._reflection = optimizable_tensors["reflection"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.densify_direction_accum = self.densify_direction_accum[valid_points_mask]
        self.densify_tangent_grad_accum = self.densify_tangent_grad_accum[valid_points_mask]
        self.densify_direction_denom = self.densify_direction_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_reflection, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "reflection": new_reflection,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._reflection = optimizable_tensors["reflection"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.enforce_parameter_bounds()

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.densify_direction_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.densify_tangent_grad_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.densify_direction_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _opacity_logits_from_values(self, opacities):
        eps = 1e-6
        return self.inverse_opacity_activation(opacities.clamp(eps, 1.0 - eps))

    @torch.no_grad()
    def _set_opacity_values(self, mask, opacities):
        if mask.numel() == 0 or not mask.any().item():
            return

        logits = self._opacity_logits_from_values(opacities).to(
            device=self._opacity.device,
            dtype=self._opacity.dtype,
        )
        self._opacity[mask] = logits

        for group in self.optimizer.param_groups:
            if group["name"] != "opacity":
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"][mask] = 0
                stored_state["exp_avg_sq"][mask] = 0
            break

    def _budget_prune_mask_by_opacity(self, max_gaussians):
        if max_gaussians is None or max_gaussians <= 0:
            return torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)

        n_points = self.get_xyz.shape[0]
        if n_points <= max_gaussians:
            return torch.zeros((n_points), device=self.get_xyz.device, dtype=torch.bool)

        prune_count = n_points - int(max_gaussians)
        opacity_values = self.get_opacity.squeeze(-1)
        prune_indices = torch.topk(
            opacity_values,
            k=prune_count,
            largest=False,
            sorted=False,
        ).indices

        prune_mask = torch.zeros((n_points), device=self.get_xyz.device, dtype=torch.bool)
        prune_mask[prune_indices] = True
        return prune_mask

    @staticmethod
    def _count_mask(mask):
        return int(mask.sum().item())

    @staticmethod
    def _format_tensor_stats(name, values):
        values = values.detach().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return f"{name}=empty"

        values = values.float()
        qs = torch.quantile(
            values,
            torch.tensor([0.5, 0.9, 0.95, 0.99], device=values.device),
        )
        return (
            f"{name}: min={values.min().item():.4g} "
            f"p50={qs[0].item():.4g} p90={qs[1].item():.4g} "
            f"p95={qs[2].item():.4g} p99={qs[3].item():.4g} "
            f"max={values.max().item():.4g} mean={values.mean().item():.4g}"
        )

    def _current_densification_directions(self):
        """Return multi-view descent directions in the current local tangent frame."""
        rotations = build_rotation(self._rotation.detach())
        local_direction_sum = torch.bmm(
            rotations.transpose(1, 2),
            self.densify_direction_accum.unsqueeze(-1),
        ).squeeze(-1)
        tangent_direction_sum = local_direction_sum[:, :2]

        observations = self.densify_direction_denom.squeeze(-1)
        tangent_sum_norm = torch.norm(tangent_direction_sum, dim=-1)
        consistency = tangent_sum_norm / torch.clamp(observations, min=1.0)
        consistency = torch.clamp(consistency, 0.0, 1.0)

        tangent_directions = tangent_direction_sum / torch.clamp(
            tangent_sum_norm[:, None],
            min=1e-12,
        )
        mean_tangent_grad = (
            self.densify_tangent_grad_accum.squeeze(-1)
            / torch.clamp(observations, min=1.0)
        )

        finite = (
            torch.isfinite(tangent_directions).all(dim=-1)
            & torch.isfinite(consistency)
            & torch.isfinite(mean_tangent_grad)
        )
        valid = (
            finite
            & (observations >= self.densify_direction_min_observations)
            & (consistency >= self.densify_direction_min_consistency)
            & (tangent_sum_norm > 1e-12)
        )

        tangent_directions[~finite] = 0.0
        consistency[~finite] = 0.0
        mean_tangent_grad[~finite] = 0.0
        return {
            "tangent_direction": tangent_directions,
            "consistency": consistency,
            "mean_tangent_grad": mean_tangent_grad,
            "observations": observations,
            "valid": valid,
        }

    @staticmethod
    def _ellipse_radius_along_direction(scales, directions):
        safe_scales = torch.clamp(scales, min=1e-12)
        inv_radius_sq = torch.sum(
            directions.square() / safe_scales.square(),
            dim=-1,
        )
        return torch.rsqrt(torch.clamp(inv_radius_sq, min=1e-12))

    def _print_densify_debug(self, stats):
        iteration = stats["iteration"]
        prefix = f"[DENSIFY iter {iteration}]" if iteration is not None else "[DENSIFY]"
        print(
            "\n"
            f"{prefix} count {stats['n_before']} -> {stats['n_after_budget']} "
            f"-> {stats['n_after_split']} -> {stats['n_after_prune']} "
            f"(budget pre-pruned {stats['prune_budget']}, "
            f"clone candidates redirected to split {stats['redirected_clone_count']}, "
            f"split selected {stats['split_count']}, "
            f"regular pruned {stats['prune_total']})",
            flush=True,
        )
        print(
            f"{prefix} thresholds grad>={stats['max_grad']:.4g}, "
            f"opacity<{stats['min_opacity']:.4g}, "
            f"former_clone_scale<={stats['scale_threshold']:.4g}, "
            f"max_screen_size={stats['max_screen_size']}, "
            f"max_gaussians={stats['max_gaussians']}",
            flush=True,
        )
        print(
            f"{prefix} visible_accum={stats['visible_accum']}/{stats['n_after_budget']} "
            f"original_split_candidates={stats['original_split_count']} "
            f"opacity_prune={stats['prune_opacity']} "
            f"big_vs_prune={stats['prune_big_vs']} "
            f"big_ws_prune={stats['prune_big_ws']} "
            f"budget_prune={stats['prune_budget']}",
            flush=True,
        )
        print(f"{prefix} {stats['grad_stats']}", flush=True)
        print(f"{prefix} {stats['opacity_stats']}", flush=True)
        print(f"{prefix} {stats['scale_stats']}", flush=True)
        print(f"{prefix} {stats['radii_stats']}", flush=True)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        if N != 2:
            raise ValueError("densify_and_split now only supports splitting into two Gaussians")

        n_init_points = self.get_xyz.shape[0]

        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_xyz = self.get_xyz[selected_pts_mask]
        selected_scaling = self.get_scaling[selected_pts_mask]
        selected_rotation = self._rotation[selected_pts_mask]

        long_axis = torch.argmax(selected_scaling, dim=1)
        long_scale = selected_scaling.gather(1, long_axis[:, None])

        rots = build_rotation(selected_rotation)
        long_dirs = torch.gather(
            rots,
            2,
            long_axis[:, None, None].expand(-1, 3, 1),
        ).squeeze(-1)
        offsets = long_dirs * (long_scale / 3.0)
        new_xyz = torch.stack(
            (selected_xyz - offsets, selected_xyz + offsets),
            dim=1,
        ).reshape(-1, 3)

        child_scaling = selected_scaling.clone()
        child_scaling.scatter_(1, long_axis[:, None], long_scale * 0.707)
        new_scaling = self.scaling_inverse_activation(
            child_scaling.repeat_interleave(N, dim=0)
        )
        new_rotation = selected_rotation.repeat_interleave(N, dim=0)
        new_reflection = self._reflection[selected_pts_mask].repeat_interleave(N, dim=0)
        new_opacity = self._opacity[selected_pts_mask].repeat_interleave(N, dim=0)

        self.densification_postfix(new_xyz, new_reflection, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * int(selected_pts_mask.sum().item()), device="cuda", dtype=torch.bool)
        ))
        self.prune_points(prune_filter)

    def densify_and_clone(
        self,
        grads,
        direction_stats,
        grad_threshold,
        scene_extent,
    ):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        cloned_opacities = self.get_opacity[selected_pts_mask] * 0.5
        self._set_opacity_values(selected_pts_mask, cloned_opacities)

        selected_xyz = self._xyz[selected_pts_mask]
        selected_scaling = self.get_scaling[selected_pts_mask]
        selected_rotation = self._rotation[selected_pts_mask]
        selected_directions = direction_stats["tangent_direction"][selected_pts_mask]
        selected_consistency = direction_stats["consistency"][selected_pts_mask]
        selected_magnitude = direction_stats["mean_tangent_grad"][selected_pts_mask]
        directed_mask = direction_stats["valid"][selected_pts_mask]

        strength = torch.zeros_like(selected_magnitude)
        if directed_mask.any().item():
            magnitude_reference = torch.quantile(
                selected_magnitude[directed_mask].float(),
                0.9,
            ).to(dtype=selected_magnitude.dtype)
            magnitude_reference = torch.clamp(magnitude_reference, min=1e-12)
            strength[directed_mask] = torch.clamp(
                selected_magnitude[directed_mask] / magnitude_reference,
                0.0,
                1.0,
            )

        offset_factor = selected_consistency * (
            self.clone_offset_min_factor
            + (self.clone_offset_max_factor - self.clone_offset_min_factor) * strength
        )
        offset_factor[~directed_mask] = 0.0
        direction_radius = torch.zeros_like(selected_magnitude)
        if directed_mask.any().item():
            direction_radius[directed_mask] = self._ellipse_radius_along_direction(
                selected_scaling[directed_mask],
                selected_directions[directed_mask],
            )
        local_offsets = selected_directions * (offset_factor * direction_radius)[:, None]
        rots = build_rotation(selected_rotation)
        world_offsets = torch.bmm(
            rots[:, :, :2],
            local_offsets.unsqueeze(-1),
        ).squeeze(-1)

        new_xyz = selected_xyz + world_offsets
        new_reflection = self._reflection[selected_pts_mask]
        new_opacities = self._opacity_logits_from_values(cloned_opacities)
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = selected_rotation

        self.densification_postfix(new_xyz, new_reflection, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        max_gaussians=None,
        debug=False,
        iteration=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[~torch.isfinite(grads)] = 0.0

        n_before = self.get_xyz.shape[0]
        budget_prune_mask = self._budget_prune_mask_by_opacity(max_gaussians)
        prune_budget = self._count_mask(budget_prune_mask)
        if budget_prune_mask.any().item():
            valid_points_mask = ~budget_prune_mask
            self.prune_points(budget_prune_mask)
            grads = grads[valid_points_mask]

        if debug:
            grad_values = torch.norm(grads, dim=-1)
            valid_grad_mask = self.denom.reshape(-1) > 0
            scale_values = self.get_scaling.max(dim=1).values
            opacity_values = self.get_opacity.squeeze()
            scale_threshold = self.percent_dense * extent
            selected_by_grad = grad_values >= max_grad
            redirected_clone_mask = torch.logical_and(
                selected_by_grad,
                scale_values <= scale_threshold,
            )
            original_split_mask = torch.logical_and(
                selected_by_grad,
                scale_values > scale_threshold,
            )
            visible_accum = self._count_mask(valid_grad_mask)
            stats = {
                "iteration": iteration,
                "n_before": n_before,
                "max_grad": float(max_grad),
                "min_opacity": float(min_opacity),
                "scale_threshold": float(scale_threshold),
                "max_screen_size": max_screen_size,
                "max_gaussians": max_gaussians,
                "visible_accum": visible_accum,
                "n_after_budget": self.get_xyz.shape[0],
                "prune_budget": prune_budget,
                "redirected_clone_count": self._count_mask(redirected_clone_mask),
                "original_split_count": self._count_mask(original_split_mask),
                "split_count": self._count_mask(selected_by_grad),
                "grad_stats": self._format_tensor_stats("grad", grad_values[valid_grad_mask]),
                "opacity_stats": self._format_tensor_stats("opacity", opacity_values),
                "scale_stats": self._format_tensor_stats("scale_max", scale_values),
                "radii_stats": self._format_tensor_stats("max_radii2D", self.max_radii2D),
            }
        else:
            stats = None

        # Clone densification is intentionally disabled. Points that previously
        # satisfied the clone scale condition are now handled by split below.
        # self.densify_and_clone(grads, direction_stats, max_grad, extent)

        self.densify_and_split(grads, max_grad, extent)
        if stats is not None:
            stats["n_after_split"] = self.get_xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if stats is not None:
            stats["prune_opacity"] = self._count_mask(prune_mask)
            stats["prune_big_vs"] = 0
            stats["prune_big_ws"] = 0

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            if stats is not None:
                stats["prune_big_vs"] = self._count_mask(big_points_vs)
                stats["prune_big_ws"] = self._count_mask(big_points_ws)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if stats is not None:
            stats["prune_total"] = self._count_mask(prune_mask)

        self.prune_points(prune_mask)
        if stats is not None:
            stats["n_after_regular_prune"] = self.get_xyz.shape[0]

        post_budget_prune_mask = self._budget_prune_mask_by_opacity(max_gaussians)
        post_budget_prune = self._count_mask(post_budget_prune_mask)
        if post_budget_prune_mask.any().item():
            self.prune_points(post_budget_prune_mask)
        if stats is not None:
            stats["prune_budget"] += post_budget_prune
            stats["prune_post_budget"] = post_budget_prune

        if stats is not None:
            stats["n_after_prune"] = self.get_xyz.shape[0]
            self._print_densify_debug(stats)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def add_densification_stats(self, screenspace_points, update_filter, position_grad=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            screenspace_points.grad[update_filter],
            dim=-1,
            keepdim=True,
        )
        self.denom[update_filter] += 1

        if position_grad is None:
            return
        if position_grad.shape != self.get_xyz.shape:
            raise ValueError(
                "position_grad must have the same shape as Gaussian positions, "
                f"got {tuple(position_grad.shape)} and {tuple(self.get_xyz.shape)}"
            )

        position_grad = position_grad.detach()
        rotations = build_rotation(self._rotation.detach())
        local_grad = torch.bmm(
            rotations.transpose(1, 2),
            position_grad.unsqueeze(-1),
        ).squeeze(-1)
        tangent_grad = local_grad[:, :2]
        tangent_magnitude = torch.norm(tangent_grad, dim=-1)
        valid_direction = (
            update_filter
            & torch.isfinite(tangent_grad).all(dim=-1)
            & torch.isfinite(tangent_magnitude)
            & (tangent_magnitude > 1e-12)
        )
        if not valid_direction.any().item():
            return

        local_descent_direction = torch.zeros_like(tangent_grad)
        local_descent_direction[valid_direction] = (
            -tangent_grad[valid_direction]
            / tangent_magnitude[valid_direction, None]
        )
        world_tangent_direction = torch.bmm(
            rotations[:, :, :2],
            local_descent_direction.unsqueeze(-1),
        ).squeeze(-1)

        self.densify_direction_accum[valid_direction] += world_tangent_direction[
            valid_direction
        ]
        self.densify_tangent_grad_accum[valid_direction] += tangent_magnitude[
            valid_direction,
            None,
        ]
        self.densify_direction_denom[valid_direction] += 1
