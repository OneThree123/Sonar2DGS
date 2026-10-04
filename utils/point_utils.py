import torch


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()

    height, width = depthmap.shape[-2:]
    device = depthmap.device
    dtype = depthmap.dtype

    c2w = c2w.to(device=device, dtype=dtype)
    full_proj_transform = view.full_proj_transform.to(device=device, dtype=dtype)

    ndc2pix = torch.tensor(
        [
            [width * 0.5, 0.0, 0.0, (width - 1) * 0.5],
            [0.0, height * 0.5, 0.0, (height - 1) * 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    ).T

    projection_matrix = c2w.T @ full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device=device, dtype=dtype),
        torch.arange(height, device=device, dtype=dtype),
        indexing="xy",
    )
    pixels = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)

    rays_d = pixels @ intrins.inverse().T @ c2w[:3, :3].T
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1, eps=1e-8)
    rays_o = c2w[:3, 3]

    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    points = depths_to_points(view, depth).reshape(*depth.shape[-2:], 3)
    output = torch.zeros_like(points)

    if points.shape[0] < 3 or points.shape[1] < 3:
        return output

    step_y = points[2:, 1:-1] - points[:-2, 1:-1]
    step_x = points[1:-1, 2:] - points[1:-1, :-2]

    normal_map = torch.nn.functional.normalize(
        torch.cross(step_y, step_x, dim=-1),
        dim=-1,
        eps=1e-6,
    )
    output[1:-1, 1:-1, :] = normal_map

    return output


def _neighbor_delta(points, valid, dim):
    center = points
    center_valid = valid
    prev_points = torch.zeros_like(points)
    next_points = torch.zeros_like(points)
    prev_valid = torch.zeros_like(valid)
    next_valid = torch.zeros_like(valid)

    if dim == 1:
        prev_points[:, 1:] = points[:, :-1]
        next_points[:, :-1] = points[:, 1:]
        prev_valid[:, 1:] = valid[:, :-1]
        next_valid[:, :-1] = valid[:, 1:]
    elif dim == 0:
        prev_points[1:] = points[:-1]
        next_points[:-1] = points[1:]
        prev_valid[1:] = valid[:-1]
        next_valid[:-1] = valid[1:]
    else:
        raise ValueError(f"dim must be 0 or 1, got {dim}")

    both = center_valid & prev_valid & next_valid
    next_only = center_valid & next_valid & ~prev_valid
    prev_only = center_valid & prev_valid & ~next_valid
    delta_valid = both | next_only | prev_only

    delta = torch.zeros_like(center)
    delta = torch.where(both[..., None], next_points - prev_points, delta)
    delta = torch.where(next_only[..., None], next_points - center, delta)
    delta = torch.where(prev_only[..., None], center - prev_points, delta)
    return delta, delta_valid


def depth_to_normal_masked(view, depth, valid_mask):
    points = depths_to_points(view, depth).reshape(*depth.shape[-2:], 3)
    output = torch.zeros_like(points)
    normal_mask = torch.zeros(points.shape[:2], device=points.device, dtype=points.dtype)

    if points.shape[0] < 3 or points.shape[1] < 3:
        return output, normal_mask

    valid = valid_mask.to(device=points.device, dtype=torch.bool)
    while valid.ndim > 2 and valid.shape[0] == 1:
        valid = valid[0]
    if valid.shape != points.shape[:2]:
        raise ValueError(
            f"valid_mask must have shape {tuple(points.shape[:2])}, got {tuple(valid.shape)}"
        )

    step_x, valid_x = _neighbor_delta(points, valid, dim=1)
    step_y, valid_y = _neighbor_delta(points, valid, dim=0)
    valid_normal = valid_x & valid_y

    normal_map = torch.nn.functional.normalize(
        torch.cross(step_y, step_x, dim=-1),
        dim=-1,
        eps=1e-6,
    )
    normal_map = torch.where(valid_normal[..., None], normal_map, torch.zeros_like(normal_map))

    output = normal_map
    normal_mask = valid_normal.to(dtype=normal_mask.dtype)
    return output, normal_mask
