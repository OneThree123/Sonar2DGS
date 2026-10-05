import unittest

import numpy as np

from utils.gaussian_mesh import sample_gaussian_surfels


class GaussianMeshSamplingTest(unittest.TestCase):
    def test_capped_sampling_preserves_existing_uncapped_protocol(self):
        count = 7
        rng = np.random.default_rng(8)
        kwargs = dict(xyz=rng.random((count, 3)), scales=rng.random((count, 2)),
                      opacity=rng.random(count), alpha_min=.099,
                      axis0=np.tile([1, 0, 0], (count, 1)), axis1=np.tile([0, 1, 0], (count, 1)),
                      normals=np.tile([0, 0, 1], (count, 1)), colors=rng.random((count, 3)),
                      samples_per_gaussian=4, radius_multiplier=1.)
        full = sample_gaussian_surfels(**kwargs)
        capped = sample_gaussian_surfels(**kwargs, max_points=13)
        indices = np.linspace(0, len(full[0]) - 1, 13).astype(np.int64)
        for before, after in zip(full, capped):
            np.testing.assert_array_equal(before[indices], after)


if __name__ == "__main__":
    unittest.main()
