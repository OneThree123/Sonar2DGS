"""Shared GS surfel sampling and Poisson export, extracted from gs2d_batch_mesh.

Keep Scene exports and quality evaluation on the same geometry protocol.
"""

import math
from importlib.metadata import version
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement

GAUSSIAN_MESH_ALPHA_MIN = 0.099


def load_open3d():
    # Open3D 0.18's NumPy vector conversion can segfault with NumPy 2.
    if version("open3d").startswith("0.18.") and int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError(
            "Open3D 0.18 requires NumPy < 2 for geometry reconstruction. "
            "Install the repository environment or run: python -m pip install 'numpy==1.26.4'"
        )
    import open3d as o3d
    return o3d

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def read_vertex_properties(ply_path: Path) -> Dict[str, np.ndarray]:
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError(f"{ply_path} does not contain a vertex element.")

    vertex = ply["vertex"].data
    props = vertex.dtype.names
    data = {name: np.asarray(vertex[name]) for name in props}
    return data


def require_fields(data: Dict[str, np.ndarray], fields: Iterable[str], ply_path: Path):
    missing = [f for f in fields if f not in data]
    if missing:
        raise ValueError(f"{ply_path} missing required fields: {missing}")


def quat_to_rotmat(q: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    q shape: [N, 4]
    order:
      - wxyz: rot_0=w, rot_1=x, rot_2=y, rot_3=z, common in 3DGS codebases
      - xyzw: rot_0=x, rot_1=y, rot_2=z, rot_3=w
    returns: [N, 3, 3]
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)

    if order == "wxyz":
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif order == "xyzw":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'.")

    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def tangent_basis_from_normals(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two tangent vectors for each normal."""
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    ref = np.zeros_like(n)
    ref[:, 2] = 1.0
    nearly_z = np.abs(np.sum(n * ref, axis=1)) > 0.9
    ref[nearly_z] = np.array([0.0, 1.0, 0.0])

    t1 = np.cross(ref, n)
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)

    t2 = np.cross(n, t1)
    t2 = t2 / np.maximum(np.linalg.norm(t2, axis=1, keepdims=True), 1e-12)

    return t1, t2


def activate_scales(data: Dict[str, np.ndarray], activation: str) -> np.ndarray:
    if "scale_0" not in data or "scale_1" not in data:
        return np.ones((len(data["x"]), 2), dtype=np.float64)

    raw = np.stack([data["scale_0"], data["scale_1"]], axis=1).astype(np.float64)

    if activation == "exp":
        # Avoid overflow. Large scales should still be filtered later.
        return np.exp(np.clip(raw, -30.0, 30.0))
    if activation == "identity":
        return raw

    raise ValueError("scale activation must be 'exp' or 'identity'.")


def activate_opacity(data: Dict[str, np.ndarray], activation: str) -> np.ndarray:
    n = len(data["x"])
    if "opacity" not in data:
        return np.ones(n, dtype=np.float64)

    raw = data["opacity"].astype(np.float64)

    if activation == "sigmoid":
        return sigmoid(raw)
    if activation == "identity":
        return raw

    raise ValueError("opacity activation must be 'sigmoid' or 'identity'.")


def scalar_to_gray(
    values: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Robustly map a scalar array to grayscale RGB."""
    v = values.astype(np.float64)
    finite = np.isfinite(v)
    if not np.any(finite):
        gray = np.full_like(v, 0.5, dtype=np.float64)
        return np.stack([gray, gray, gray], axis=1)

    lo, hi = np.percentile(v[finite], [low_percentile, high_percentile])
    if hi <= lo:
        gray = np.full_like(v, 0.5, dtype=np.float64)
    else:
        gray = np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    if gamma != 1.0:
        gray = np.power(np.clip(gray, 0.0, 1.0), gamma)

    return np.stack([gray, gray, gray], axis=1)


def make_colors(
    data: Dict[str, np.ndarray],
    mask: np.ndarray,
    xyz: np.ndarray,
    normals: np.ndarray,
    opacity: np.ndarray,
    mode: str,
    reflection_low: float,
    reflection_high: float,
    color_gamma: float,
    constant_gray: float,
) -> Optional[np.ndarray]:
    """
    Make RGB colors for visualization.

    Important:
    - For sonar / learned 2DGS, reflection can be very noisy or high-frequency.
      In that case, use --color-mode constant, normal, height, or opacity.
    """
    n = xyz.shape[0]

    if mode == "none":
        return None

    if mode == "constant":
        g = float(np.clip(constant_gray, 0.0, 1.0))
        return np.tile(np.array([[g, g, g]], dtype=np.float64), (n, 1))

    if mode == "normal":
        # Normal visualization. Values in [-1, 1] are mapped to [0, 1].
        return np.clip(normals * 0.5 + 0.5, 0.0, 1.0)

    if mode == "height":
        # Use z coordinate as a smooth pseudo-color in grayscale.
        return scalar_to_gray(
            xyz[:, 2],
            low_percentile=1.0,
            high_percentile=99.0,
            gamma=color_gamma,
        )

    if mode == "opacity":
        return scalar_to_gray(
            opacity,
            low_percentile=1.0,
            high_percentile=99.0,
            gamma=color_gamma,
        )

    if mode == "reflection":
        if "reflection" not in data:
            return np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float64), (n, 1))

        return scalar_to_gray(
            data["reflection"][mask],
            low_percentile=reflection_low,
            high_percentile=reflection_high,
            gamma=color_gamma,
        )

    raise ValueError(f"Unknown color mode: {mode}")


def filter_gaussians(
    data: Dict[str, np.ndarray],
    opacity: np.ndarray,
    scales: np.ndarray,
    alpha_min: float,
    scale_min: float,
    scale_max: Optional[float],
    scale_percentile: Optional[float],
) -> np.ndarray:
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    radius_max = np.max(scales, axis=1)

    mask = np.all(np.isfinite(xyz), axis=1)
    mask &= np.all(np.isfinite(scales), axis=1)
    mask &= np.isfinite(opacity)
    mask &= opacity >= alpha_min
    mask &= np.min(scales, axis=1) >= scale_min

    if scale_max is not None:
        mask &= radius_max <= scale_max

    if scale_percentile is not None:
        valid_r = radius_max[mask]
        if len(valid_r) > 0:
            cap = np.percentile(valid_r, scale_percentile)
            mask &= radius_max <= cap

    return mask


def build_oriented_gaussian_arrays(
    data: Dict[str, np.ndarray],
    mask: np.ndarray,
    scale_activation: str,
    opacity_activation: str,
    quat_order: str,
    normal_source: str,
):
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)[mask]
    scales_all = activate_scales(data, scale_activation)
    scales = scales_all[mask]
    opacity = activate_opacity(data, opacity_activation)[mask]

    n = xyz.shape[0]

    has_normals = all(k in data for k in ["nx", "ny", "nz"])
    has_rot = all(k in data for k in ["rot_0", "rot_1", "rot_2", "rot_3"])

    if has_rot:
        q = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float64)[mask]
        R = quat_to_rotmat(q, quat_order)
        axis0 = R[:, :, 0]
        axis1 = R[:, :, 1]
        normal_from_rot = R[:, :, 2]
    else:
        R = None
        axis0 = None
        axis1 = None
        normal_from_rot = None

    if has_normals:
        normals_ply = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float64)[mask]
        normals_ply = normals_ply / np.maximum(np.linalg.norm(normals_ply, axis=1, keepdims=True), 1e-12)
    else:
        normals_ply = None

    if normal_source == "ply" and normals_ply is not None:
        normals = normals_ply
    elif normal_source == "rot" and normal_from_rot is not None:
        normals = normal_from_rot
    elif normals_ply is not None:
        normals = normals_ply
    elif normal_from_rot is not None:
        normals = normal_from_rot
    else:
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (n, 1))

    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    if axis0 is None or axis1 is None:
        axis0, axis1 = tangent_basis_from_normals(normals)
    else:
        axis0 = axis0 / np.maximum(np.linalg.norm(axis0, axis=1, keepdims=True), 1e-12)
        axis1 = axis1 / np.maximum(np.linalg.norm(axis1, axis=1, keepdims=True), 1e-12)

    return xyz, scales, opacity, axis0, axis1, normals


def deterministic_standard_normal_2d(samples_per_gaussian: int) -> np.ndarray:
    """
    Deterministic 2D standard-normal samples.

    The radial CDF of a 2D isotropic Gaussian is Rayleigh:
      F(r) = 1 - exp(-r^2 / 2)
    so r = sqrt(-2 log(1 - u)).  Using stratified quantiles keeps sampling
    reproducible while following the Gaussian density instead of a flat disk.
    returns [K, 2]
    """
    if samples_per_gaussian <= 0:
        return np.empty((0, 2), dtype=np.float64)
    if samples_per_gaussian == 1:
        return np.array([[0.0, 0.0]], dtype=np.float64)

    pts = []
    if samples_per_gaussian % 2 == 1:
        pts.append([0.0, 0.0])

    pair_count = samples_per_gaussian // 2
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(pair_count):
        u = (i + 0.5) / pair_count
        r = math.sqrt(-2.0 * math.log(max(1.0 - u, 1e-12)))
        if pair_count == 1:
            theta = math.pi / 4.0
        elif pair_count == 2:
            theta = (i + 0.5) * math.pi / pair_count
        else:
            theta = i * golden
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        pts.append([x, y])
        pts.append([-x, -y])
    return np.asarray(pts, dtype=np.float64)


def sample_gaussian_surfels(
    xyz: np.ndarray,
    scales: np.ndarray,
    opacity: np.ndarray,
    alpha_min: float,
    axis0: np.ndarray,
    axis1: np.ndarray,
    normals: np.ndarray,
    colors: Optional[np.ndarray],
    samples_per_gaussian: int,
    radius_multiplier: float,
    max_points: Optional[int] = None,
):
    if alpha_min <= 0:
        raise ValueError(f"alpha_min must be positive, got {alpha_min}")

    sample_counts = np.floor(opacity / alpha_min * samples_per_gaussian + 0.5).astype(np.int64)
    sample_counts = np.maximum(sample_counts, 1)
    cumulative = np.cumsum(sample_counts)
    total_points = int(cumulative[-1]) if len(cumulative) else 0
    # Preserve the original linspace downsampling exactly without allocating
    # all uncapped samples (which may require gigabytes for a large model).
    if max_points is not None and total_points > max_points:
        selected = np.linspace(0, total_points - 1, max_points).astype(np.int64)
    else:
        selected = np.arange(total_points, dtype=np.int64)
    gaussian_ids = np.searchsorted(cumulative, selected, side="right")
    starts = np.concatenate(([0], cumulative[:-1]))
    local_ids = selected - starts[gaussian_ids]
    points = np.empty((len(selected), 3), dtype=np.float64)
    sampled_normals = np.empty_like(points)
    sampled_colors = None if colors is None else np.empty_like(points)

    ids, offsets, counts = np.unique(gaussian_ids, return_index=True, return_counts=True)
    for i, offset, count in zip(ids, offsets, counts):
        next_offset = offset + count
        gaussian_samples = deterministic_standard_normal_2d(int(sample_counts[i]))[
            local_ids[offset:next_offset]]
        u, v = gaussian_samples[:, 0], gaussian_samples[:, 1]
        points[offset:next_offset] = (
            xyz[i][None, :]
            + radius_multiplier * scales[i, 0] * u[:, None] * axis0[i][None, :]
            + radius_multiplier * scales[i, 1] * v[:, None] * axis1[i][None, :]
        )
        sampled_normals[offset:next_offset] = normals[i][None, :]
        if sampled_colors is not None:
            sampled_colors[offset:next_offset] = colors[i][None, :]

    return points, sampled_normals, sampled_colors


def write_ply_pointcloud(path: Path, points: np.ndarray, normals: Optional[np.ndarray] = None, colors: Optional[np.ndarray] = None):
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
    ]

    if normals is not None:
        arrays += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]

    if colors is not None:
        arrays += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    vertex = np.empty(len(points), dtype=arrays)
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]

    if normals is not None:
        vertex["nx"] = normals[:, 0]
        vertex["ny"] = normals[:, 1]
        vertex["nz"] = normals[:, 2]

    if colors is not None:
        rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        vertex["red"] = rgb[:, 0]
        vertex["green"] = rgb[:, 1]
        vertex["blue"] = rgb[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def build_surfel_ellipse_mesh(
    xyz: np.ndarray,
    scales: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    colors: np.ndarray,
    radius_multiplier: float,
    ellipse_segments: int,
    max_gaussians: Optional[int] = None,
):
    n = xyz.shape[0]
    if max_gaussians is not None and n > max_gaussians:
        idx = np.linspace(0, n - 1, max_gaussians).astype(np.int64)
        xyz = xyz[idx]
        scales = scales[idx]
        axis0 = axis0[idx]
        axis1 = axis1[idx]
        colors = colors[idx]
        n = len(idx)

    angles = np.linspace(0, 2.0 * np.pi, ellipse_segments, endpoint=False)
    ca = np.cos(angles)
    sa = np.sin(angles)

    # For each Gaussian: one center + ring vertices.
    vertices = []
    faces = []
    vcolors = []

    for i in range(n):
        base = len(vertices)
        vertices.append(xyz[i])
        vcolors.append(colors[i])

        ring = (
            xyz[i][None, :]
            + radius_multiplier * scales[i, 0] * ca[:, None] * axis0[i][None, :]
            + radius_multiplier * scales[i, 1] * sa[:, None] * axis1[i][None, :]
        )
        vertices.extend(ring)
        vcolors.extend(np.repeat(colors[i][None, :], ellipse_segments, axis=0))

        for j in range(ellipse_segments):
            a = base
            b = base + 1 + j
            c = base + 1 + ((j + 1) % ellipse_segments)
            faces.append([a, b, c])

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32), np.asarray(vcolors, dtype=np.float64)


def write_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: Optional[np.ndarray] = None):
    path.parent.mkdir(parents=True, exist_ok=True)

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    v = np.empty(len(vertices), dtype=dtype)
    v["x"] = vertices[:, 0]
    v["y"] = vertices[:, 1]
    v["z"] = vertices[:, 2]

    if colors is not None:
        rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        v["red"] = rgb[:, 0]
        v["green"] = rgb[:, 1]
        v["blue"] = rgb[:, 2]

    f = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    f["vertex_indices"] = faces

    PlyData(
        [
            PlyElement.describe(v, "vertex"),
            PlyElement.describe(f, "face"),
        ],
        text=False,
    ).write(str(path))


def run_poisson_reconstruction(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    colors: Optional[np.ndarray],
    out_mesh_path: Path,
    depth: int,
    density_quantile: float,
    remove_outlier: bool,
    outlier_nb_neighbors: int,
    outlier_std_ratio: float,
):
    o3d = load_open3d()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    if remove_outlier and len(points) > outlier_nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors,
            std_ratio=outlier_std_ratio,
        )

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
        )

    # Make normals more consistent. Useful for Poisson.
    try:
        pcd.orient_normals_consistent_tangent_plane(30)
    except Exception:
        # Some very sparse or disconnected point clouds may fail here.
        pass

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=False,
    )

    densities = np.asarray(densities)
    if density_quantile > 0.0:
        threshold = np.quantile(densities, density_quantile)
        remove_mask = densities < threshold
        mesh.remove_vertices_by_mask(remove_mask)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    out_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    if not mesh.has_triangles():
        raise RuntimeError("Poisson reconstruction produced no valid triangles")
    if not o3d.io.write_triangle_mesh(str(out_mesh_path), mesh):
        raise OSError(f"Could not write Poisson mesh: {out_mesh_path}")


def _normalize_optional_limit(value: Optional[int]) -> Optional[int]:
    if value is not None and value < 0:
        return None
    return value


def _normalize_normals(normals: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if normals is None:
        return None

    normals = np.asarray(normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = np.all(np.isfinite(normals), axis=1) & (lengths[:, 0] > 1e-12)
    if not np.any(valid):
        return None

    normalized = np.zeros_like(normals)
    normalized[valid] = normals[valid] / lengths[valid]
    normalized[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normalized


def _sample_filtered_gaussian_surfels_from_ply(
    ply_path,
    *,
    opacity_activation: str = "sigmoid",
    scale_activation: str = "exp",
    quat_order: str = "wxyz",
    normal_source: str = "auto",
    color_mode: str = "none",
    constant_gray: float = 0.72,
    reflection_low: float = 5.0,
    reflection_high: float = 95.0,
    color_gamma: float = 1.0,
    alpha_min: float = 0.10,
    scale_min: float = 1e-8,
    scale_max: Optional[float] = None,
    scale_percentile: Optional[float] = 99.0,
    samples_per_gaussian: int = 4,
    radius_multiplier: float = 1.0,
    max_points: Optional[int] = 600000,
):
    ply_path = Path(ply_path)
    max_points = _normalize_optional_limit(max_points)

    data = read_vertex_properties(ply_path)
    require_fields(data, ["x", "y", "z"], ply_path)

    scales_all = activate_scales(data, scale_activation)
    opacity_all = activate_opacity(data, opacity_activation)

    mask = filter_gaussians(
        data=data,
        opacity=opacity_all,
        scales=scales_all,
        alpha_min=alpha_min,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_percentile=scale_percentile,
    )

    if int(mask.sum()) == 0:
        raise RuntimeError(f"All gaussians were filtered out: {ply_path}")

    xyz, scales, opacity, axis0, axis1, normals = build_oriented_gaussian_arrays(
        data=data,
        mask=mask,
        scale_activation=scale_activation,
        opacity_activation=opacity_activation,
        quat_order=quat_order,
        normal_source=normal_source,
    )

    colors = make_colors(
        data=data,
        mask=mask,
        xyz=xyz,
        normals=normals,
        opacity=opacity,
        mode=color_mode,
        reflection_low=reflection_low,
        reflection_high=reflection_high,
        color_gamma=color_gamma,
        constant_gray=constant_gray,
    )

    pts, nrm, col = sample_gaussian_surfels(
        xyz=xyz,
        scales=scales,
        opacity=opacity,
        alpha_min=alpha_min,
        axis0=axis0,
        axis1=axis1,
        normals=normals,
        colors=colors,
        samples_per_gaussian=samples_per_gaussian,
        radius_multiplier=radius_multiplier,
        max_points=max_points,
    )

    return data, mask, pts, nrm, col


def export_sampled_pointcloud_from_gaussian_ply(
    ply_path,
    out_pointcloud_path,
    *,
    opacity_activation: str = "sigmoid",
    scale_activation: str = "exp",
    quat_order: str = "wxyz",
    normal_source: str = "auto",
    color_mode: str = "none",
    constant_gray: float = 0.72,
    reflection_low: float = 5.0,
    reflection_high: float = 95.0,
    color_gamma: float = 1.0,
    alpha_min: float = 0.10,
    scale_min: float = 1e-8,
    scale_max: Optional[float] = None,
    scale_percentile: Optional[float] = 99.0,
    samples_per_gaussian: int = 4,
    radius_multiplier: float = 1.0,
    max_points: Optional[int] = 600000,
):
    """Export filtered surfel samples from a 2DGS Gaussian PLY without meshing."""
    out_pointcloud_path = Path(out_pointcloud_path)
    data, mask, pts, nrm, col = _sample_filtered_gaussian_surfels_from_ply(
        ply_path,
        opacity_activation=opacity_activation,
        scale_activation=scale_activation,
        quat_order=quat_order,
        normal_source=normal_source,
        color_mode=color_mode,
        constant_gray=constant_gray,
        reflection_low=reflection_low,
        reflection_high=reflection_high,
        color_gamma=color_gamma,
        alpha_min=alpha_min,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_percentile=scale_percentile,
        samples_per_gaussian=samples_per_gaussian,
        radius_multiplier=radius_multiplier,
        max_points=max_points,
    )

    write_ply_pointcloud(out_pointcloud_path, pts, normals=nrm, colors=col)

    return {
        "input": str(ply_path),
        "output": str(out_pointcloud_path),
        "num_input": len(data["x"]),
        "num_kept": int(mask.sum()),
        "num_sampled_points": int(len(pts)),
    }


def export_poisson_mesh_from_gaussian_ply(
    ply_path,
    out_mesh_path,
    *,
    opacity_activation: str = "sigmoid",
    scale_activation: str = "exp",
    quat_order: str = "wxyz",
    normal_source: str = "auto",
    color_mode: str = "constant",
    constant_gray: float = 0.72,
    reflection_low: float = 5.0,
    reflection_high: float = 95.0,
    color_gamma: float = 1.0,
    alpha_min: float = 0.10,
    scale_min: float = 1e-8,
    scale_max: Optional[float] = None,
    scale_percentile: Optional[float] = 99.0,
    samples_per_gaussian: int = 4,
    radius_multiplier: float = 1.0,
    max_points: Optional[int] = 600000,
    poisson_depth: int = 9,
    density_quantile: float = 0.02,
    remove_outlier: bool = False,
    outlier_nb_neighbors: int = 30,
    outlier_std_ratio: float = 2.0,
    sampled_pointcloud_path=None,
):
    """Export one Poisson mesh from a 2DGS Gaussian PLY file."""
    ply_path = Path(ply_path)
    out_mesh_path = Path(out_mesh_path)

    data, mask, pts, nrm, col = _sample_filtered_gaussian_surfels_from_ply(
        ply_path,
        opacity_activation=opacity_activation,
        scale_activation=scale_activation,
        quat_order=quat_order,
        normal_source=normal_source,
        color_mode=color_mode,
        constant_gray=constant_gray,
        reflection_low=reflection_low,
        reflection_high=reflection_high,
        color_gamma=color_gamma,
        alpha_min=alpha_min,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_percentile=scale_percentile,
        samples_per_gaussian=samples_per_gaussian,
        radius_multiplier=radius_multiplier,
        max_points=max_points,
    )

    if sampled_pointcloud_path is not None:
        write_ply_pointcloud(Path(sampled_pointcloud_path), pts, normals=nrm, colors=col)

    run_poisson_reconstruction(
        points=pts,
        normals=nrm,
        colors=col,
        out_mesh_path=out_mesh_path,
        depth=poisson_depth,
        density_quantile=density_quantile,
        remove_outlier=remove_outlier,
        outlier_nb_neighbors=outlier_nb_neighbors,
        outlier_std_ratio=outlier_std_ratio,
    )

    return {
        "input": str(ply_path),
        "output": str(out_mesh_path),
        "num_input": len(data["x"]),
        "num_kept": int(mask.sum()),
        "num_poisson_points": int(len(pts)),
    }


def export_poisson_mesh_from_pointcloud_ply(
    ply_path,
    out_mesh_path,
    *,
    color_mode: str = "constant",
    constant_gray: float = 0.72,
    reflection_low: float = 5.0,
    reflection_high: float = 95.0,
    color_gamma: float = 1.0,
    max_points: Optional[int] = 600000,
    poisson_depth: int = 9,
    density_quantile: float = 0.02,
    remove_outlier: bool = False,
    outlier_nb_neighbors: int = 30,
    outlier_std_ratio: float = 2.0,
):
    """Export one Poisson mesh from a regular point-cloud PLY file."""
    ply_path = Path(ply_path)
    out_mesh_path = Path(out_mesh_path)
    max_points = _normalize_optional_limit(max_points)

    data = read_vertex_properties(ply_path)
    require_fields(data, ["x", "y", "z"], ply_path)

    xyz_all = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    mask = np.all(np.isfinite(xyz_all), axis=1)
    if int(mask.sum()) == 0:
        raise RuntimeError(f"No finite points found: {ply_path}")

    xyz = xyz_all[mask]
    normals = None
    if all(k in data for k in ["nx", "ny", "nz"]):
        normals_all = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float64)
        normals = _normalize_normals(normals_all[mask])

    color_normals = normals
    if color_normals is None:
        color_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (xyz.shape[0], 1))

    colors = make_colors(
        data=data,
        mask=mask,
        xyz=xyz,
        normals=color_normals,
        opacity=np.ones(xyz.shape[0], dtype=np.float64),
        mode=color_mode,
        reflection_low=reflection_low,
        reflection_high=reflection_high,
        color_gamma=color_gamma,
        constant_gray=constant_gray,
    )

    if max_points is not None and len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        xyz = xyz[idx]
        if normals is not None:
            normals = normals[idx]
        if colors is not None:
            colors = colors[idx]

    run_poisson_reconstruction(
        points=xyz,
        normals=normals,
        colors=colors,
        out_mesh_path=out_mesh_path,
        depth=poisson_depth,
        density_quantile=density_quantile,
        remove_outlier=remove_outlier,
        outlier_nb_neighbors=outlier_nb_neighbors,
        outlier_std_ratio=outlier_std_ratio,
    )

    return {
        "input": str(ply_path),
        "output": str(out_mesh_path),
        "num_input": len(data["x"]),
        "num_kept": int(mask.sum()),
        "num_poisson_points": int(len(xyz)),
    }
