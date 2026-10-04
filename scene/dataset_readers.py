#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import json
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from utils.graphics_utils import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    C2W: np.ndarray

    FovY: float
    FovX: float
    range_max: float

    image: Image.Image
    mask: Optional[Image.Image]
    image_path: str
    image_name: str

    tx_height: int
    tx_width: int
    img_height: int
    img_width: int
    normalizing_method: str
    data_type: str


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: Optional[str]


def _random_unit_normals(count):
    normals = np.random.normal(size=(count, 3)).astype(np.float32)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    bad = lengths[:, 0] < 1e-8
    if np.any(bad):
        normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        lengths[bad] = 1.0
    return normals / lengths


def _stratified_uniform_points(bbox_min, bbox_max, count):
    """Sample one random point per approximately cube-shaped, equal-volume stratum."""
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)

    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        raise ValueError("Point-cloud bounding box must contain two 3D coordinates.")
    if not np.all(np.isfinite(bbox_min)) or not np.all(np.isfinite(bbox_max)):
        raise ValueError("Point-cloud bounding box contains non-finite values.")
    if np.any(bbox_max < bbox_min):
        raise ValueError("Point-cloud bounding-box maximum is smaller than its minimum.")

    count = int(count)
    if count <= 0:
        raise ValueError(f"Point count must be positive, got {count}.")

    points = np.empty((count, 3), dtype=np.float64)
    strata = [(bbox_min, bbox_max, 0, count)]

    while strata:
        lower, upper, start, stratum_count = strata.pop()
        if stratum_count == 1:
            points[start] = np.random.uniform(lower, upper)
            continue

        left_count = stratum_count // 2
        right_count = stratum_count - left_count
        split_axis = int(np.argmax(upper - lower))
        split_position = (
            lower[split_axis]
            + (upper[split_axis] - lower[split_axis])
            * left_count
            / stratum_count
        )

        left_upper = upper.copy()
        left_upper[split_axis] = split_position
        right_lower = lower.copy()
        right_lower[split_axis] = split_position

        strata.append((right_lower, upper, start + left_count, right_count))
        strata.append((lower, left_upper, start, left_count))

    np.random.shuffle(points)
    return points.astype(np.float32)


def _normalize_or_randomize_normals(normals, path):
    normals = np.asarray(normals, dtype=np.float32)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    finite = np.all(np.isfinite(normals), axis=1)
    valid = finite & (lengths[:, 0] > 1e-8)

    normalized = np.zeros_like(normals, dtype=np.float32)
    normalized[valid] = normals[valid] / lengths[valid]

    invalid_count = int(np.count_nonzero(~valid))
    if invalid_count > 0:
        normalized[~valid] = _random_unit_normals(invalid_count)
        print(
            f"Point cloud {path} has {invalid_count} invalid normals; "
            "replaced them with random unit normals."
        )

    return normalized


def getNerfppNorm(cam_info):
    if len(cam_info) == 0:
        raise ValueError("Cannot compute scene normalization without cameras.")

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        cam_centers.append(cam.C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    if radius < 1e-6:
        radius = 1.0

    return {"translate": -center, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    names = vertices.data.dtype.names

    required_fields = ("x", "y", "z")
    missing_fields = [name for name in required_fields if name not in names]
    if missing_fields:
        raise ValueError(f"PLY file {path} is missing fields: {missing_fields}")

    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    if "reflection" in names:
        reflections = np.asarray(vertices["reflection"], dtype=np.float32)
    else:
        reflections = np.ones(positions.shape[0], dtype=np.float32)
        print(f"PLY file {path} has no reflection; generated reflection=1 for all points.")

    if all(name in names for name in ("nx", "ny", "nz")):
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T.astype(np.float32)
        normals = _normalize_or_randomize_normals(normals, path)
    else:
        normals = _random_unit_normals(positions.shape[0])
        print(f"PLY file {path} has no normals; generated random unit normals.")

    return BasicPointCloud(points=positions, reflections=reflections, normals=normals)


def storePly(path, point_cloud):
    path = Path(path)
    positions = np.asarray(point_cloud.points, dtype=np.float32)
    reflections = np.asarray(point_cloud.reflections, dtype=np.float32).reshape(-1)
    normals = np.asarray(point_cloud.normals, dtype=np.float32)

    if normals.shape != positions.shape:
        raise ValueError(
            f"Point cloud normals must have shape {positions.shape}, got {normals.shape}."
        )
    if reflections.shape[0] != positions.shape[0]:
        raise ValueError(
            f"Point cloud reflections must have {positions.shape[0]} values, "
            f"got {reflections.shape[0]}."
        )

    dtype_full = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("reflection", "f4"),
    ]
    elements = np.empty(positions.shape[0], dtype=dtype_full)
    elements["x"] = positions[:, 0]
    elements["y"] = positions[:, 1]
    elements["z"] = positions[:, 2]
    elements["nx"] = normals[:, 0]
    elements["ny"] = normals[:, 1]
    elements["nz"] = normals[:, 2]
    elements["reflection"] = reflections

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def readBasicInfo(path):
    path = Path(path)
    info_path = path / "basic_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"basic_info.json not found: {info_path}")

    with open(info_path, "r") as json_file:
        info = json.load(json_file)

    required_fields = (
        "img_width",
        "img_height",
        "tx_width",
        "tx_height",
        "range_max",
        "fovx",
        "fovy",
    )
    missing_fields = [field for field in required_fields if field not in info]
    if missing_fields:
        raise ValueError(f"{info_path} is missing fields: {missing_fields}")

    basic_info = {
        "img_width": int(info["img_width"]),
        "img_height": int(info["img_height"]),
        "tx_width": int(info["tx_width"]),
        "tx_height": int(info["tx_height"]),
        "range_max": float(info["range_max"]),
        "fovx": float(info["fovx"]),
        "fovy": float(info["fovy"]),
        "normalizing_method": str(info.get("normalizing_method", "all")).lower(),
        "data_type": str(info.get("data_type", "")).upper(),
        "has_mask": bool(info.get("has_mask", False)),
    }

    if basic_info["normalizing_method"] not in ("all", "range"):
        raise ValueError(
            f"normalizing_method must be 'all' or 'range', "
            f"got {basic_info['normalizing_method']!r}"
        )

    for key in ("img_width", "img_height", "tx_width", "tx_height"):
        if basic_info[key] <= 0:
            raise ValueError(f"{key} must be positive, got {basic_info[key]}")

    for key in ("range_max", "fovx", "fovy"):
        if basic_info[key] <= 0:
            raise ValueError(f"{key} must be positive, got {basic_info[key]}")

    return basic_info


def readSonarCameras(path, split, basic_info):
    root = Path(path)
    pose_path = root / split / "pose.json"
    if not pose_path.exists():
        raise FileNotFoundError(f"{split} pose file not found: {pose_path}")

    with open(pose_path, "r") as json_file:
        pose_data = json.load(json_file)

    if isinstance(pose_data, list):
        frames = pose_data
    elif isinstance(pose_data, dict) and "frames" in pose_data:
        frames = pose_data["frames"]
    else:
        raise ValueError(f"{pose_path} must contain a list or a dict with a 'frames' list.")

    for frame_idx, frame in enumerate(frames):
        if "file_path" not in frame:
            raise KeyError(f"{pose_path} frame {frame_idx} is missing 'file_path'.")

    frames = sorted(
        frames,
        key=lambda frame: (
            0,
            int(Path(frame["file_path"]).stem),
        )
        if Path(frame["file_path"]).stem.isdigit()
        else (
            1,
            Path(frame["file_path"]).as_posix(),
        ),
    )

    cam_infos = []
    for idx, frame in enumerate(frames):
        rel_image_path = Path(frame["file_path"])
        image_path = root / rel_image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if "c2w" in frame:
            c2w_data = frame["c2w"]
        elif "C2W" in frame:
            c2w_data = frame["C2W"]
        elif "transform_matrix" in frame:
            c2w_data = frame["transform_matrix"]
        else:
            raise KeyError(
                f"{pose_path} frame {idx} must contain one of: 'c2w', 'C2W', 'transform_matrix'."
            )

        c2w = np.asarray(c2w_data, dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(f"{pose_path} frame {idx} C2W must have shape [4, 4], got {c2w.shape}.")

        if not np.all(np.isfinite(c2w)):
            raise ValueError(f"{pose_path} frame {idx} C2W contains non-finite values.")

        if not np.allclose(c2w[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-5):
            raise ValueError(f"{pose_path} frame {idx} C2W last row must be [0, 0, 0, 1].")

        image = Image.open(image_path).convert("L")
        real_w, real_h = image.size
        if real_w != basic_info["img_width"] or real_h != basic_info["img_height"]:
            raise ValueError(
                f"Image size mismatch for {image_path}: "
                f"file=({real_w}, {real_h}), "
                f"basic_info=({basic_info['img_width']}, {basic_info['img_height']})"
            )

        mask = Image.new("L", image.size, 255)
        if basic_info["has_mask"] and "mask_path" in frame:
            mask_path = root / Path(frame["mask_path"])
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            mask = Image.open(mask_path).convert("L")
            mask_w, mask_h = mask.size
            if mask_w != basic_info["img_width"] or mask_h != basic_info["img_height"]:
                raise ValueError(
                    f"Mask size mismatch for {mask_path}: "
                    f"file=({mask_w}, {mask_h}), "
                    f"basic_info=({basic_info['img_width']}, {basic_info['img_height']})"
                )
        elif basic_info["has_mask"] and split == "train":
            raise KeyError(f"{pose_path} frame {idx} is missing 'mask_path'.")

        cam_infos.append(
            CameraInfo(
                uid=idx,
                C2W=c2w,
                FovY=basic_info["fovy"],
                FovX=basic_info["fovx"],
                range_max=basic_info["range_max"],
                image=image,
                mask=mask,
                image_path=str(image_path),
                image_name=image_path.stem,
                tx_height=basic_info["tx_height"],
                tx_width=basic_info["tx_width"],
                img_height=basic_info["img_height"],
                img_width=basic_info["img_width"],
                normalizing_method=basic_info["normalizing_method"],
                data_type=basic_info["data_type"],
            )
        )

    return cam_infos


def createSonarFrustumRandomPointCloud(train_cam_infos, basic_info, count=100):
    if len(train_cam_infos) == 0:
        raise ValueError("Cannot initialize sonar point cloud without training cameras.")

    count = int(count)
    if count <= 0:
        raise ValueError(f"point_cloud_init_count must be positive, got {count}.")

    range_max = float(basic_info["range_max"])
    fovx = float(basic_info["fovx"])
    fovy = float(basic_info["fovy"])
    half_width = range_max * np.tan(0.5 * fovx)
    half_height = range_max * np.tan(0.5 * fovy)

    local_frustum_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [range_max, -half_width, -half_height],
            [range_max, -half_width, half_height],
            [range_max, half_width, -half_height],
            [range_max, half_width, half_height],
        ],
        dtype=np.float32,
    )

    frustum_points = []
    for cam_info in train_cam_infos:
        c2w = cam_info.C2W.astype(np.float32, copy=False)
        rotation = c2w[:3, :3]
        translation = c2w[:3, 3]
        frustum_points.append(local_frustum_points @ rotation.T + translation)

    frustum_points = np.concatenate(frustum_points, axis=0).astype(np.float32)
    if not np.all(np.isfinite(frustum_points)):
        raise ValueError("Sonar frustum points contain non-finite values.")

    bbox_min = frustum_points.min(axis=0)
    bbox_max = frustum_points.max(axis=0)
    points = _stratified_uniform_points(bbox_min, bbox_max, count)
    normals = _random_unit_normals(count)
    reflections = np.ones(count, dtype=np.float32)

    print(
        f"Generated {count} stratified-random sonar frustum points from "
        f"{len(train_cam_infos)} training poses inside bbox "
        f"min={bbox_min.tolist()}, max={bbox_max.tolist()}."
    )
    return BasicPointCloud(points=points, reflections=reflections, normals=normals)


def readSonarSceneInfo(path, point_cloud_init="auto", point_cloud_init_count=100):
    root = Path(path)
    basic_info = readBasicInfo(root)

    print("Reading sonar training poses")
    train_cam_infos = readSonarCameras(root, "train", basic_info)
    print("Reading sonar testing poses")
    test_cam_infos = readSonarCameras(root, "test", basic_info)

    if len(train_cam_infos) == 0:
        raise ValueError("No training cameras were loaded.")

    point_cloud_init = str(point_cloud_init).lower()
    valid_init_methods = ("auto", "ply", "sonar_frustum_random")
    if point_cloud_init not in valid_init_methods:
        raise ValueError(
            f"point_cloud_init must be one of {valid_init_methods}, got {point_cloud_init!r}."
        )

    ply_path = root / "points3d.ply"
    if point_cloud_init == "sonar_frustum_random":
        pcd = createSonarFrustumRandomPointCloud(
            train_cam_infos,
            basic_info,
            point_cloud_init_count,
        )
        ply_path = None
    elif ply_path.exists():
        pcd = fetchPly(ply_path)
    elif point_cloud_init == "auto":
        print(
            f"Initial sonar point cloud not found at {ply_path}; "
            "using sonar_frustum_random initialization."
        )
        pcd = createSonarFrustumRandomPointCloud(
            train_cam_infos,
            basic_info,
            point_cloud_init_count,
        )
        ply_path = None
    else:
        raise FileNotFoundError(f"Initial sonar point cloud not found: {ply_path}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=str(ply_path) if ply_path is not None else None,
    )


sceneLoadTypeCallbacks = {
    "Sonar": readSonarSceneInfo,
}
