"""Poisson reference caching and predicted mesh-to-reference surface distance."""

import hashlib
import json
import os
import time
import uuid
from importlib.metadata import version
from pathlib import Path

import numpy as np

from utils.gaussian_mesh import (
    GAUSSIAN_MESH_ALPHA_MIN,
    load_open3d,
    export_poisson_mesh_from_gaussian_ply,
    export_poisson_mesh_from_pointcloud_ply,
)


POISSON_CONFIG = dict(max_points=600000, poisson_depth=9, density_quantile=0.02,
                      remove_outlier=False, outlier_nb_neighbors=30, outlier_std_ratio=2.0)
PREDICTED_CONFIG = dict(
    **POISSON_CONFIG, alpha_min=GAUSSIAN_MESH_ALPHA_MIN, scale_min=1e-8,
    scale_max=None, scale_percentile=99.0, samples_per_gaussian=4, radius_multiplier=1.0,
    opacity_activation="sigmoid", scale_activation="exp", quat_order="wxyz",
    normal_source="auto", color_mode="constant", constant_gray=0.72,
)
SURFACE_CONFIG = dict(sample_count=100000, seed=0, chunk_size=32768)
RECONSTRUCTION_PROTOCOL = dict(
    version=1, normal_knn=30, orient_knn=30, poisson_width=0, poisson_scale=1.1,
    linear_fit=False, cleanup=["duplicate_vertices", "duplicate_triangles",
                             "degenerate_triangles", "non_manifold_edges"],
)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_mesh(path):
    o3d = load_open3d()

    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh_arrays(mesh)
    return mesh


def mesh_arrays(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Mesh has no valid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Mesh has no valid triangles")
    if not np.isfinite(vertices).all() or (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError("Mesh has non-finite coordinates or invalid triangle indices")
    return vertices, faces


def sample_mesh_surface(mesh, count=SURFACE_CONFIG["sample_count"], seed=SURFACE_CONFIG["seed"]):
    """Area-weighted surface samples with a local RNG, independent of training RNGs."""
    if count <= 0:
        raise ValueError("Surface sample count must be positive")
    vertices, faces = mesh_arrays(mesh)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    area_sum = areas.sum()
    if not np.isfinite(area_sum) or area_sum <= 0:
        raise ValueError("Mesh has no positive-area surface")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(faces), count, p=areas / area_sum)]
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    return ((1 - root) * selected[:, 0] + root * (1 - uv[:, 1:]) * selected[:, 1]
            + root * uv[:, 1:] * selected[:, 2])


class MeshSurfaceDistance:
    """Cache Open3D's triangle acceleration structure on CPU; no rtree dependency."""

    def __init__(self, mesh):
        o3d = load_open3d()

        mesh_arrays(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def distances(self, points, chunk_size=SURFACE_CONFIG["chunk_size"]):
        o3d = load_open3d()

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("Geometry samples must be a nonempty finite N x 3 array")
        if chunk_size <= 0:
            raise ValueError("Distance chunk size must be positive")
        chunks = []
        for start in range(0, len(points), chunk_size):
            query = o3d.core.Tensor(points[start:start + chunk_size].astype(np.float32))
            chunks.append(self.scene.compute_distance(query).numpy().astype(np.float64))
        distances = np.concatenate(chunks)
        if not np.isfinite(distances).all():
            raise ValueError("Non-finite point-to-triangle distance")
        return distances


class GeometryEvaluator:
    def __init__(self, dataset_path, model_path):
        started = time.perf_counter()
        source = Path(dataset_path).resolve() / "dense_points3d.ply"
        if not source.is_file():
            raise FileNotFoundError(f"Quality evaluation requires dataset point cloud: {source}")
        reference_config = dict(source=str(source), source_sha256=file_digest(source),
                                poisson=POISSON_CONFIG, protocol=RECONSTRUCTION_PROTOCOL,
                                open3d_version=version("open3d"), coordinates="dataset_world", unit="m")
        signature = config_digest(reference_config)
        directory = Path(model_path) / "quality_time_tests" / "reference" / signature
        directory.mkdir(parents=True, exist_ok=True)
        self.reference_path = directory / "gt_poisson_mesh.ply"
        metadata_path = directory / "metadata.json"
        cached = False
        if self.reference_path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text())
                cached = (metadata["config"] == reference_config
                          and metadata["mesh_sha256"] == file_digest(self.reference_path))
                if cached:
                    read_mesh(self.reference_path)
            except (ValueError, KeyError):
                cached = False
        if not cached:
            temporary = directory / ("gt_" + uuid.uuid4().hex + ".ply")
            try:
                export_poisson_mesh_from_pointcloud_ply(source, temporary, **POISSON_CONFIG)
                read_mesh(temporary)
                os.replace(temporary, self.reference_path)
                write_json_atomic(metadata_path, dict(config=reference_config,
                                  mesh_sha256=file_digest(self.reference_path),
                                  reference_setup_seconds=time.perf_counter() - started))
            finally:
                temporary.unlink(missing_ok=True)
        self.distance = MeshSurfaceDistance(read_mesh(self.reference_path))
        self.setup_seconds = time.perf_counter() - started
        self.config = dict(reference_signature=signature, reference_mesh=str(self.reference_path.resolve()),
                           reference=reference_config, predicted=PREDICTED_CONFIG,
                           surface=SURFACE_CONFIG, metric="predicted_mesh_surface_to_gt_mesh_surface_mean", unit="m")
        print(f"[Quality geometry] {'Loaded cached' if cached else 'Built'} GT: {self.reference_path}")

    def evaluate(self, gaussians, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gaussian_path = output_dir / "gaussian.ply"
        sampled_path = output_dir / "sampled_surface_points.ply"
        mesh_path = output_dir / "pred_poisson_mesh.ply"
        result = dict(p2s=None, p2s_unit="m", num_geometry_samples=0,
                      num_poisson_input_points=0, num_pred_mesh_vertices=0, num_pred_mesh_faces=0,
                      geometry_status="failed", p2s_status="failed", pred_mesh_path=None, error=None)
        artifacts = None
        gaussians.save_ply(str(gaussian_path))
        try:
            info = export_poisson_mesh_from_gaussian_ply(
                gaussian_path, mesh_path, sampled_pointcloud_path=sampled_path, **PREDICTED_CONFIG)
            mesh = read_mesh(mesh_path)
            vertices, faces = mesh_arrays(mesh)
            result.update(num_poisson_input_points=info["num_poisson_points"],
                          num_pred_mesh_vertices=len(vertices), num_pred_mesh_faces=len(faces),
                          geometry_status="ok", pred_mesh_path=str(mesh_path.resolve()))
            artifacts = dict(sampled_points_path=str(sampled_path), pred_mesh_path=str(mesh_path))
            samples = sample_mesh_surface(mesh)
            result.update(p2s=float(self.distance.distances(samples).mean()), p2s_status="ok",
                          num_geometry_samples=len(samples))
        except (ValueError, RuntimeError) as exc:
            if "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                raise
            result["error"] = str(exc)
        return result, artifacts
