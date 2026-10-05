import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from plyfile import PlyData, PlyElement

from utils import geometry_metrics as metrics
from utils.gaussian_mesh import load_open3d, write_ply_pointcloud


class GeometryMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o3d = load_open3d()

    def plane(self, height=0):
        mesh = self.o3d.geometry.TriangleMesh()
        mesh.vertices = self.o3d.utility.Vector3dVector([[0, 0, height], [2, 0, height], [0, 2, height]])
        mesh.triangles = self.o3d.utility.Vector3iVector([[0, 1, 2]])
        return mesh

    def test_distance_to_triangle_interior_and_normal_offset(self):
        distance = metrics.MeshSurfaceDistance(self.plane())
        points = np.array([[.5, .5, 0], [.5, .5, .25], [.5, .5, -.4]])
        np.testing.assert_allclose(distance.distances(points, 1), [0, .25, .4], atol=1e-6)
        np.testing.assert_allclose(distance.distances(points, 1), distance.distances(points, 100))
        with self.assertRaises(ValueError):
            distance.distances([])

    def test_mesh_surface_protocol_is_deterministic_and_does_not_change_rng(self):
        before = np.random.get_state()
        samples = metrics.sample_mesh_surface(self.plane(.3), count=1000)
        np.testing.assert_array_equal(samples, metrics.sample_mesh_surface(self.plane(.3), count=1000))
        np.testing.assert_array_equal(before[1], np.random.get_state()[1])
        distances = metrics.MeshSurfaceDistance(self.plane()).distances(samples)
        np.testing.assert_allclose(distances, .3, atol=1e-6)

    def test_missing_reference_fails_before_evaluation(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileNotFoundError):
            metrics.GeometryEvaluator(root, root)

    def test_real_poisson_cache_and_gs_mesh_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(10)
            normals = rng.normal(size=(700, 3))
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            write_ply_pointcloud(root / "dense_points3d.ply", normals, normals=normals)
            poisson = dict(metrics.POISSON_CONFIG, poisson_depth=5)
            predicted = dict(metrics.PREDICTED_CONFIG, poisson_depth=5)
            export = metrics.export_poisson_mesh_from_pointcloud_ply
            with patch.object(metrics, "POISSON_CONFIG", poisson), patch.object(
                metrics, "PREDICTED_CONFIG", predicted,
            ), patch.object(metrics, "export_poisson_mesh_from_pointcloud_ply", wraps=export) as builder:
                evaluator = metrics.GeometryEvaluator(root, root)
                self.assertEqual(builder.call_count, 1)
                resumed = metrics.GeometryEvaluator(root, root)
                self.assertEqual(builder.call_count, 1)
                self.assertEqual(evaluator.reference_path, resumed.reference_path)

                class GS:
                    def __init__(self, opacity=0.5):
                        self.opacity = opacity

                    def save_ply(self, path):
                        names = ["x", "y", "z", "nx", "ny", "nz", "scale_0", "scale_1", "opacity"]
                        values = np.zeros(len(normals), dtype=[(key, "f4") for key in names])
                        for i, key in enumerate(["x", "y", "z"]):
                            values[key] = normals[:, i]
                        for i, key in enumerate(["nx", "ny", "nz"]):
                            values[key] = normals[:, i]
                        values["scale_0"] = values["scale_1"] = np.log(.03)
                        values["opacity"] = np.log(self.opacity / (1 - self.opacity))
                        PlyData([PlyElement.describe(values, "vertex")]).write(path)

                result, artifacts = evaluator.evaluate(GS(), root / "prediction")
                self.assertEqual(result["geometry_status"], "ok", result["error"])
                self.assertEqual(result["p2s_status"], "ok", result["error"])
                self.assertTrue(np.isfinite(result["p2s"]))
                self.assertEqual(result["p2s_unit"], "m")
                self.assertEqual(result["num_geometry_samples"], 100000)
                self.assertGreater(result["num_pred_mesh_faces"], 0)
                self.assertTrue(Path(artifacts["sampled_points_path"]).exists())
                failed, artifacts = evaluator.evaluate(GS(.01), root / "filtered")
                self.assertIsNone(failed["p2s"])
                self.assertEqual(failed["geometry_status"], "failed")
                self.assertIsNone(artifacts)


if __name__ == "__main__":
    unittest.main()
