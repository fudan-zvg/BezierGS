#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
import random
from matplotlib import cm
import open3d as o3d


def save_ply(points, filename, rgbs=None, loading_bar=True, max_points=1e10):
    if type(points) in [torch.Tensor, torch.nn.parameter.Parameter]:
        points = points.detach().cpu().numpy()
    if type(rgbs) in [torch.Tensor, torch.nn.parameter.Parameter]:
        rgbs = rgbs.detach().cpu().numpy()

    if rgbs is None:
        rgbs = np.ones_like(points[:, [0]])
    if rgbs.shape[1] == 1:
        colormap = cm.get_cmap('turbo')
        rgbs = colormap(rgbs[:, 0])[:, :3]
    
    # 确保颜色值在0-1范围内
    rgbs = np.clip(rgbs, 0, 1)

    pcd = o3d.geometry.PointCloud()

    # 将 xyz 和 rgb 数据添加到点云中
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)  # 将 RGB 转换到 [0, 1] 范围
    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd)

def load_ply(filename):
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)  # 点的坐标 (Nx3)
    colors = np.asarray(pcd.colors)  # 点的颜色 (Nx3)
    return points, colors


def GridSample3D(in_pc, in_shs, voxel_size=0.013):
    in_pc_ = in_pc[:, :3].copy()
    quantized_pc = np.around(in_pc_ / voxel_size)
    quantized_pc -= np.min(quantized_pc, axis=0)
    pc_boundary = np.max(quantized_pc, axis=0) - np.min(quantized_pc, axis=0)

    voxel_index = quantized_pc[:, 0] * pc_boundary[1] * pc_boundary[2] + quantized_pc[:, 1] * pc_boundary[2] + quantized_pc[:, 2]

    split_point, index = get_split_point(voxel_index)

    in_points = in_pc[index, :]
    out_points = in_points[split_point[:-1], :]

    in_colors = in_shs[index]
    out_colors = in_colors[split_point[:-1]]

    return out_points, out_colors


def get_split_point(labels):
    index = np.argsort(labels)
    label = labels[index]
    label_shift = label.copy()

    label_shift[1:] = label[:-1]
    remain = label - label_shift
    step_index = np.where(remain > 0)[0].tolist()
    step_index.insert(0, 0)
    step_index.append(labels.shape[0])
    return step_index, index


def sample_on_aabb_surface(aabb_center, aabb_size, n_pts=1000, above_half=False):
    """
    0:立方体的左面(x轴负方向)
    1:立方体的右面(x轴正方向)
    2:立方体的下面(y轴负方向)
    3:立方体的上面(y轴正方向)
    4:立方体的后面(z轴负方向)
    5:立方体的前面(z轴正方向)
    """
    # Choose a face randomly
    faces = np.random.randint(0, 6, size=n_pts)

    # Generate two random numbers
    r_ = np.random.random((n_pts, 2))

    # Create an array to store the points
    points = np.zeros((n_pts, 3))

    # Define the offsets for each face
    offsets = np.array([
        [-aabb_size[0] / 2, 0, 0],
        [aabb_size[0] / 2, 0, 0],
        [0, -aabb_size[1] / 2, 0],
        [0, aabb_size[1] / 2, 0],
        [0, 0, -aabb_size[2] / 2],
        [0, 0, aabb_size[2] / 2]
    ])

    # Define the scales for each face
    scales = np.array([
        [aabb_size[1], aabb_size[2]],
        [aabb_size[1], aabb_size[2]],
        [aabb_size[0], aabb_size[2]],
        [aabb_size[0], aabb_size[2]],
        [aabb_size[0], aabb_size[1]],
        [aabb_size[0], aabb_size[1]]
    ])

    # Define the positions of the zero column for each face
    zero_column_positions = [0, 0, 1, 1, 2, 2]
    # Define the indices of the aabb_size components for each face
    aabb_size_indices = [[1, 2], [1, 2], [0, 2], [0, 2], [0, 1], [0, 1]]
    # Calculate the coordinates of the points for each face
    for i in range(6):
        mask = faces == i
        r_scaled = r_[mask] * scales[i]
        r_scaled = np.insert(r_scaled, zero_column_positions[i], 0, axis=1)
        aabb_size_adjusted = np.insert(aabb_size[aabb_size_indices[i]] / 2, zero_column_positions[i], 0)
        points[mask] = aabb_center + offsets[i] + r_scaled - aabb_size_adjusted
        # visualize_points(points[mask], aabb_center, aabb_size)
    # visualize_points(points, aabb_center, aabb_size)

    # 提取上半部分的点
    if above_half:
        points = points[points[:, -1] > aabb_center[-1]]
    return points


def get_OccGrid(pts, aabb, occ_voxel_size):
    # 计算网格的大小
    grid_size = np.ceil((aabb[1] - aabb[0]) / occ_voxel_size).astype(int)
    assert pts.min() >= aabb[0].min() and pts.max() <= aabb[1].max(), "Points are outside the AABB"

    # 创建一个空的网格
    voxel_grid = np.zeros(grid_size, dtype=np.uint8)

    # 将点云转换为网格坐标
    grid_pts = ((pts - aabb[0]) / occ_voxel_size).astype(int)

    # 将网格中的点设置为1
    voxel_grid[grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2]] = 1

    # check
    # voxel_coords = np.floor((pts - aabb[0]) / occ_voxel_size).astype(int)
    # occ = voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]

    return voxel_grid


def visualize_depth(depth, near=0.2, far=13, linear=False):
    depth = depth[0].clone().detach().cpu().numpy()
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    if linear:
        curve_fn = lambda x: -x
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.) * 255
    out_depth = torch.from_numpy(out_depth).permute(2, 0, 1).float().cuda() / 255
    return out_depth


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_step_lr_func(lr_init, lr_final, start_step):
    def helper(step):
        if step < start_step:
            return lr_init
        else:
            return lr_final

    return helper


def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotation_to_quaternion(R):
    r11, r12, r13 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r21, r22, r23 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r31, r32, r33 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    qw = torch.sqrt((1 + r11 + r22 + r33).clamp_min(1e-7)) / 2
    qx = (r32 - r23) / (4 * qw)
    qy = (r13 - r31) / (4 * qw)
    qz = (r21 - r12) / (4 * qw)

    quaternion = torch.stack((qw, qx, qy, qz), dim=-1)
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    return quaternion


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    r11 = 1 - 2 * y * y - 2 * z * z
    r12 = 2 * x * y - 2 * w * z
    r13 = 2 * x * z + 2 * w * y

    r21 = 2 * x * y + 2 * w * z
    r22 = 1 - 2 * x * x - 2 * z * z
    r23 = 2 * y * z - 2 * w * x

    r31 = 2 * x * z - 2 * w * y
    r32 = 2 * y * z + 2 * w * x
    r33 = 1 - 2 * x * x - 2 * y * y

    rotation_matrix = torch.stack((torch.stack((r11, r12, r13), dim=1),
                                   torch.stack((r21, r22, r23), dim=1),
                                   torch.stack((r31, r32, r33), dim=1)), dim=1)
    return rotation_matrix


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result_quaternion = torch.stack((w, x, y, z), dim=1)
    return result_quaternion


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import logging

import sys


def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][%(levelname)s] - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)

    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
