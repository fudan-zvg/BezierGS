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
import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_step_lr_func
from torch import nn
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB, IDFT
from simple_knn._C import distCUDA2
from utils.system_utils import mkdir_p
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, save_ply
from utils.general_utils import rotation_to_quaternion, quaternion_multiply, quaternion_to_rotation_matrix
import torch.nn.functional as F
from scene.cameras import Camera
import logging
from scipy.special import comb as n_over_k
from scene.deform_model import DeformModel

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)

        self.BezierCoeff = self.get_bezier_coefficient()
        self.BezierDerivativeCoeff = self.get_d_bezier_coefficient()

    def __init__(self, args):
        self.active_sh_degree = 0
        self.fourier_dim = args.fourier_dim
        self.num_control_points = args.order + 1 # new number of control points
        self.max_sh_degree = args.sh_degree
        self._control_points = torch.empty(0) # new control points
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._group = torch.empty(0) # new group_num, int type

        self.dynamic_mode = args.dynamic_mode # Bezier curve, PVG sin, DeformableGS MLP
        self.deform = DeformModel(False)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.time_duration = args.time_duration
        self.frame_interval_cnt = args.end_time - args.start_time
        self.no_time_split = args.no_time_split
        self.contract = args.contract
        self.t_init = args.t_init
        self.big_point_threshold = args.big_point_threshold

        self.opa_init = args.opa_init

        # self.velocity_decay = args.velocity_decay # 1.0
        self.random_init_point = args.random_init_point # 200000

        self.binomial_coefs = torch.tensor(
            [math.comb(self.num_control_points - 1, k) for k in range(self.num_control_points)],
            dtype=torch.float32,
        ).cuda()

        self.group_num = 0
        self.g_id = 0
        self.world_bezier_time_map = dict()
        self.trajectory_dict = dict()
        self.trajectory_cp_tensor = None
        self.cp_recorded_timestamp_2_bezier_t = None
        self.exist_range = dict()

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._control_points,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._group,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args=None):
        (self.active_sh_degree,
            self._control_points,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._group,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.setup_functions()
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._control_points[:, 0, :]
    
    def get_feats(self, time=None):
        if time is None:
            return self.get_features
        else:
            return self.get_features4d(time)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_features4d(self, time):  # 4DSH
        all_features = []
        for idx in range(self.group_num + 1):
            if idx == 0:
                normalized_time = time
            else:
                normalized_time = (time - self.exist_range[idx][0]) / (self.exist_range[idx][1] - self.exist_range[idx][0])
            
            mask = self.get_group == idx
            idft_base = IDFT(normalized_time, self.fourier_dim)[0].cuda()
            features_dc = self._features_dc[mask] # [N, C, 3]
            features_dc = torch.sum(features_dc * idft_base[..., None], dim=1, keepdim=True) # [N, 1, 3]
            features_rest = self._features_rest[mask] # [N, sh, 3]
            features = torch.cat([features_dc, features_rest], dim=1) # [N, (sh + 1) * C, 3]
            all_features.append(features)
        return torch.cat(all_features, dim=0)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_control_points(self):
        return self._control_points
    
    @property
    def get_group(self):
        return self._group

    def get_xyz_with_offset(self, world_time):
        if self.dynamic_mode == "Bezier":
            valid_mask = torch.zeros(self._control_points.shape[0], dtype=torch.bool, device="cuda")
            valid_mask[self.get_group == 0] = True
            xyz = self._control_points[:, 0, :].clone()
            xyz_offset = torch.zeros_like(xyz)

            for idx in range(1, self.group_num + 1):
                if world_time < self.exist_range[idx][0] or world_time > self.exist_range[idx][1]:
                    continue
                
                # calculate bezier parameter
                bezier_param_cp = self.cp_recorded_timestamp_2_bezier_t[idx-1]
                bezier_param = self.BezierCoeff(torch.tensor([[world_time]]).cuda()) @ bezier_param_cp

                traj_cp = self.trajectory_cp_tensor[idx-1]
                M = self.BezierCoeff(bezier_param).to(dtype=traj_cp.dtype, device=traj_cp.device)
                traj_central = torch.matmul(M, traj_cp).squeeze(0)

                xyz_offset[self.get_group == idx] = self.predict_xyz(bezier_param, idx).squeeze(0)
                xyz[self.get_group == idx] = xyz_offset[self.get_group == idx] + traj_central
                valid_mask[self.get_group == idx] = True
        elif self.dynamic_mode == "PVG":
            pass
        elif self.dynamic_mode == "DeformableGS":
            xyz = self._control_points[:, 0, :].clone()
            valid_mask = torch.ones(self._control_points.shape[0], dtype=torch.bool, device="cuda")
            xyz_offset = torch.zeros_like(xyz)
            dyn_mask = self.get_group > 0
            dyn_xyz = xyz[dyn_mask]
            world_time = torch.tensor([[world_time]], dtype=torch.float32, device="cuda").expand(dyn_xyz.shape[0], 1)
            d_xyz, _, _ = self.deform.step(dyn_xyz.detach(), world_time)
            xyz[dyn_mask] = dyn_xyz + d_xyz

        return xyz, xyz_offset, valid_mask
    
    def get_inst_velocity(self, world_time):
        if self.dynamic_mode == "Bezier":
            velocity = torch.zeros_like(self._control_points[:, 0, :])
            valid_mask = torch.zeros(self._control_points.shape[0], dtype=torch.bool, device="cuda")
            valid_mask[self.get_group == 0] = True

            for idx in range(1, self.group_num + 1):
                if world_time < self.exist_range[idx][0] or world_time > self.exist_range[idx][1]:
                    continue

                bezier_param_cp = self.cp_recorded_timestamp_2_bezier_t[idx-1]
                bezier_param = self.BezierCoeff(torch.tensor([[world_time]]).cuda()) @ bezier_param_cp

                velocity[self.get_group == idx] = self.predict_xyz_derivative(bezier_param, idx).squeeze(0)
                valid_mask[self.get_group == idx] = True
        elif self.dynamic_mode == "PVG":
            pass
        elif self.dynamic_mode == "DeformableGS":
            velocity = torch.zeros_like(self._control_points[:, 0, :])
            valid_mask = torch.zeros(self._control_points.shape[0], dtype=torch.bool, device="cuda")

        return velocity, valid_mask
    
    def get_rotation_matrix(self):
        return quaternion_to_rotation_matrix(self.get_rotation)
    
    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1) ** 2

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_ply_dict(self, ply_dict, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        gaussian_control_point = []
        gaussian_color = []
        gaussian_scales = []
        gaussian_group = []

        obj_num = len(ply_dict.keys()) - 1 # 1 is background
        self.trajectory_cp_tensor = torch.zeros([obj_num, 4, 3], dtype=torch.float32).cuda()
        self.cp_recorded_timestamp_2_bezier_t = torch.zeros([obj_num, 4, 1], dtype=torch.float32).cuda()

        for k, v in ply_dict.items():
            if k == 'bkgd':
                bg_xyz = torch.from_numpy(v['xyz_array']).float().cuda()
                bg_color = torch.from_numpy(v['colors_array']).float().cuda()
                control_points = bg_xyz
                control_points = control_points.unsqueeze(1)
                control_points = control_points.repeat(1, 4, 1)

                # sphere init
                r_max = 100000
                r_min = 2
                num_sph = self.random_init_point

                theta = 2*torch.pi*torch.rand(num_sph)
                phi = (torch.pi/2*0.99*torch.rand(num_sph))**1.5 # x**a decay
                s = torch.rand(num_sph)
                r_1 = s*1/r_min+(1-s)*1/r_max
                r = 1/r_1
                pts_sph = torch.stack([r*torch.cos(theta)*torch.cos(phi), r*torch.sin(theta)*torch.cos(phi), r*torch.sin(phi)],dim=-1).cuda()
                pts_sph = pts_sph.unsqueeze(1).repeat(1, 4, 1)
                pts_sph[:, :, 2] = -pts_sph[:, :, 2]+1
                control_points = torch.cat([control_points, pts_sph], dim=0)
                bg_color = torch.cat([
                    bg_color,
                    torch.zeros(pts_sph.shape[0], 3).float().cuda()
                ], dim=0)

                gaussian_control_point.append(control_points)
                gaussian_color.append(bg_color)
                # get dist2
                dist2 = torch.clamp_min(distCUDA2(control_points[:, 0, :]), 0.0000001)
                scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
                gaussian_scales.append(scales)

                gaussian_group.append(torch.zeros((control_points.shape[0]), dtype=torch.int))
            else:
                # lidar may not cover the object or skip the sparse object
                if v['xyz_offset'] is None or v['xyz_offset'].shape[0] < 30:
                    continue
                if self.dynamic_mode == "Bezier":
                    self.group_num += 1
                    xyz_offset = torch.from_numpy(v['xyz_offset']).float().cuda()
                    obj_color = torch.from_numpy(v['colors_array']).float().cuda()
                    trajectory = torch.from_numpy(v['trajectory']).float().cuda()
                    t_parameterized = self.get_chord_len_parametrization(trajectory)
                    trajectory_cp = self.generate_control_points(trajectory.unsqueeze(0), t_parameterized.unsqueeze(1)).squeeze(0) # [num_control_points, 3]

                    for round in range(100):
                        t_samples = torch.linspace(0, 1, 10000, dtype=trajectory.dtype, device=trajectory.device)
                        Bernstein_mat = self.BezierCoeff(t_samples.unsqueeze(1)) # [num_samples, num_control_points]
                        curve_samples = Bernstein_mat @ trajectory_cp # [num_samples, 3]
                        diff = trajectory.unsqueeze(1) - curve_samples.unsqueeze(0) # [num_frames, num_samples, 3]
                        diff_sq = torch.sum(diff**2, dim=-1) # [num_frames, num_samples]
                        indices = torch.argmin(diff_sq, dim=-1) # [num_frames]

                        bezier_t = t_samples[indices]
                        new_cp = self.generate_control_points(trajectory.unsqueeze(0), bezier_t.unsqueeze(1)).squeeze(0) # [num_control_points, 3]

                        diff_ctrl = new_cp - trajectory_cp
                        trajectory_cp = new_cp
                        if torch.norm(diff_ctrl, dim=-1).max() < 1e-6:
                            print("round: ", round)
                            break
                        if round == 99:
                            print("round99: ", round)

                    self.trajectory_dict[self.group_num] = dict(zip(v['timestamp_list'], [torch.tensor(pos, device='cuda') for pos in trajectory.tolist()]))
                    self.trajectory_cp_tensor[self.group_num-1] = trajectory_cp
                    self.world_bezier_time_map[self.group_num] = dict(zip(v['timestamp_list'], [torch.tensor([[t]], device='cuda') for t in bezier_t.tolist()]))
                    self.cp_recorded_timestamp_2_bezier_t[self.group_num-1] = torch.linalg.lstsq(self.BezierCoeff(torch.tensor(v['timestamp_list']).unsqueeze(1).cuda()), bezier_t.unsqueeze(1).cuda()).solution
                    self.exist_range[self.group_num] = [v['timestamp_list'][0], v['timestamp_list'][-1]]

                    offset_cp = self.generate_control_points(xyz_offset, bezier_t.unsqueeze(1)).squeeze(0)
                    gaussian_control_point.append(offset_cp)
                    gaussian_color.append(obj_color)

                    dist2 = torch.clamp_min(distCUDA2(offset_cp[:, 0, :]), 0.0000001)
                    scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
                    gaussian_scales.append(scales)

                    gaussian_group.append(torch.ones((offset_cp.shape[0]), dtype=torch.int) * self.group_num)
                elif self.dynamic_mode == "PVG":
                    pass
                elif self.dynamic_mode == "DeformableGS":
                    self.group_num += 1
                    mid_idx = v['xyz_offset'].shape[1] // 2
                    xyz_offset = torch.from_numpy(v['xyz_offset'][:, mid_idx, :]).float().cuda()
                    obj_color = torch.from_numpy(v['colors_array']).float().cuda()
                    trajectory = torch.from_numpy(v['trajectory'][mid_idx, :]).float().cuda()

                    xyz_pos = xyz_offset + trajectory
                    gaussian_control_point.append(xyz_pos.unsqueeze(1).repeat(1, 4, 1))
                    gaussian_color.append(obj_color)

                    dist2 = torch.clamp_min(distCUDA2(xyz_pos), 0.0000001)
                    scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
                    gaussian_scales.append(scales)
                    gaussian_group.append(torch.ones((xyz_offset.shape[0]), dtype=torch.int) * self.group_num)
                    
        gaussian_control_point = torch.cat(gaussian_control_point, dim=0)
        gaussian_color = torch.cat(gaussian_color, dim=0)
        gaussian_scales = torch.cat(gaussian_scales, dim=0)
        gaussian_group = torch.cat(gaussian_group, dim=0).cuda()

        fused_color = RGB2SH(gaussian_color)
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        
        logging.info("Number of points at initialization: {}".format(gaussian_control_point.shape[0]))

        rots = torch.zeros((gaussian_control_point.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(self.opa_init * torch.ones((gaussian_control_point.shape[0], 1), dtype=torch.float, device="cuda"))

        self._control_points = nn.Parameter(gaussian_control_point.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(gaussian_scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._group = gaussian_group.requires_grad_(False)
        self.max_radii2D = torch.zeros((self.get_control_points.shape[0]), device="cuda")

        self.trajectory_cp_tensor = nn.Parameter(self.trajectory_cp_tensor, requires_grad=True)
        self.cp_recorded_timestamp_2_bezier_t = nn.Parameter(self.cp_recorded_timestamp_2_bezier_t, requires_grad=True)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_control_points.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_control_points.shape[0], 1), device="cuda")

        l = [
            {'params': [self._control_points], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "control_points"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.cp_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)
        
        bezier_l = [
            {'params': self.trajectory_cp_tensor, 'lr': training_args.traj_pos_lr_init * self.spatial_lr_scale, "name": "traj"},
            {'params': self.cp_recorded_timestamp_2_bezier_t, 'lr': training_args.t_lr, "name": "bezier_param"}
        ]
        self.bezier_optimizer = torch.optim.Adam(bezier_l, lr=0.0, eps=1e-15)
        self.traj_cp_scheduler_args = get_expon_lr_func(lr_init=training_args.traj_pos_lr_init * self.spatial_lr_scale,
                                                        lr_final=training_args.traj_pos_lr_final * self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.iterations)

        if self.dynamic_mode == "DeformableGS":
            self.deform.train_setting(training_args)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "control_points":
                lr = self.cp_scheduler_args(iteration)
                param_group['lr'] = lr
        for param_group in self.bezier_optimizer.param_groups:
            if param_group["name"] == "traj":
                lr = self.traj_cp_scheduler_args(iteration)
                param_group['lr'] = lr
        if self.dynamic_mode == "DeformableGS":
            self.deform.update_learning_rate(iteration)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * self.opa_init))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._control_points = optimizable_tensors["control_points"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._group = self._group[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_control_points, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"control_points": new_control_points,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._control_points = optimizable_tensors["control_points"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_control_points.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_control_points.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_control_points.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_control_points.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        if self.contract:
            scale_factor = self._control_points[:, 0, :].norm(dim=-1)*scene_extent-1 # -0
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/scene_extent
        else:
            scale_factor = torch.ones_like(self._control_points)[:, 0, :]/scene_extent

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent*scale_factor)
        decay_factor = N*0.8

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda") # (N, 3)
        samples = torch.normal(mean=means, std=stds) # (N, 3)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1) # (N, 3, 3)
        control_points = self.get_control_points[selected_pts_mask] # (N, 4, 3)

        new_control_points = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1).unsqueeze(1) + control_points.repeat(N, 1, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (decay_factor))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_control_points, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        new_group = self._group[selected_pts_mask].repeat(N)
        self._group = torch.cat((self._group, new_group), dim=0)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        if self.contract:
            scale_factor = self._control_points[:, 0, :].norm(dim=-1)*scene_extent-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/scene_extent
        else:
            scale_factor = torch.ones_like(self._control_points)[:, 0, :]/scene_extent

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling,dim=1).values <= self.percent_dense * scene_extent*scale_factor)

        new_control_points = self._control_points[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_control_points, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        new_group = self._group[selected_pts_mask]
        self._group = torch.cat((self._group, new_group), dim=0)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if self.contract:
            scale_factor = self._control_points[:, 0, :].norm(dim=-1)*extent-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/extent
        else:
            scale_factor = torch.ones_like(self._control_points)[:, 0, :]/extent

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > self.big_point_threshold * extent * scale_factor  ## ori 0.1
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
        
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def get_points_depth_in_depth_map(self, fov_camera : Camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0) # 0
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.fx / points_in_camera_space[:,2] + fov_camera.cx,
                         points_in_camera_space[:,1] * fov_camera.fy / points_in_camera_space[:,2] + fov_camera.cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera : Camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        return pts


    def get_bezier_coefficient(self):
        # ts: (T, 1) -> M: (T, num_control_points)
        n = self.num_control_points - 1
        ks = torch.arange(self.num_control_points, dtype=torch.float32, device="cuda")

        def bezier_coeff(ts: torch.Tensor):
            t_pow_k = ts ** ks # (T, num_cp)
            one_minus_t_pow_n_minus_k = (1.0 - ts) ** (n - ks)
            M = self.binomial_coefs * t_pow_k * one_minus_t_pow_n_minus_k # (T, num_cp)
            return M
        
        return bezier_coeff

    def get_d_bezier_coefficient(self):
        # ts: (T, 1) -> dM: (T, 1, num_control_points)
        n = self.num_control_points - 1
        ks = torch.arange(self.num_control_points, dtype=torch.float32, device="cuda")
        binomial_coefs_2d = self.binomial_coefs.unsqueeze(0)  # (1, num_control_points)
        ks_2d = ks.unsqueeze(0)                               # (1, num_control_points)
        n_minus_ks_2d = (n - ks).unsqueeze(0)                 # (1, num_control_points)

        def d_bezier_coeff(ts: torch.Tensor):
            one_minus_ts = (1.0 - ts)

            ts_pow_k_minus_1 = torch.zeros(ts.shape[0], self.num_control_points, device="cuda") # [N, num_cp]
            ts_pow_k_minus_1[:, 1:] = ts ** (ks_2d[:, 1:] - 1) # [T, num_cp - 1]
            
            one_minus_ts_pow_n_minus_k = one_minus_ts ** n_minus_ks_2d # [T, num_cp]
            term1 = binomial_coefs_2d * ks_2d * ts_pow_k_minus_1 * one_minus_ts_pow_n_minus_k # [T, num_cp]

            one_minus_ts_pow_n_minus_k_minus_1 = torch.zeros(ts.shape[0], self.num_control_points, device="cuda")
            one_minus_ts_pow_n_minus_k_minus_1[:, :-1] = one_minus_ts ** (n_minus_ks_2d[:, :-1] - 1)

            ts_pow_k = ts ** ks_2d
            term2 = binomial_coefs_2d * n_minus_ks_2d * ts_pow_k * one_minus_ts_pow_n_minus_k_minus_1
            dM = term1 - term2

            return dM.unsqueeze(1) # [T, 1, num_cp]

        return d_bezier_coeff

    def generate_control_points(self, xyz_array, time_list):
        # xyz_array: [N, num_frames, 3]
        # time_list: [num_frames, 1]
        # Return: control_points: [N, num_control_points, 3]
        M = self.BezierCoeff(time_list).to(dtype=xyz_array.dtype, device=xyz_array.device).unsqueeze(0)
        MT = M.transpose(1, 2)
        MTM = torch.matmul(MT, M)
        MTM_inv = torch.linalg.pinv(MTM)
        control_points = torch.matmul(MTM_inv, torch.matmul(MT, xyz_array))
        return control_points
    
    def predict_xyz(self, t, group_idx):
        M = self.BezierCoeff(t).to(dtype=self._control_points.dtype, device=self._control_points.device).unsqueeze(1)
        return torch.matmul(M, self._control_points[self._group == group_idx]).squeeze(1)
    
    def predict_xyz_derivative(self, t, group_idx):
        d_M = self.BezierDerivativeCoeff(t).to(dtype=self._control_points.dtype, device=self._control_points.device)
        return torch.matmul(d_M, self._control_points[self._group == group_idx]).squeeze(1)

    def get_chord_len_parametrization(self, trajectory):
        # trajectory: [num_frames, 3]
        num_frames, _ = trajectory.shape
        if num_frames == 1:
            return torch.zeros((1), dtype=trajectory.dtype, device=trajectory.device)
        
        dist = torch.norm(trajectory[1:] - trajectory[:-1], dim=1) # [num_frames - 1]
        s_0 = torch.zeros(1, dtype=trajectory.dtype, device=trajectory.device)
        s = torch.cat([s_0, torch.cumsum(dist, dim=0)]) # [num_frames]

        total_len = s[-1]
        eps = 1e-12

        if total_len < eps:
            t = torch.linspace(0, 1, num_frames, dtype=trajectory.dtype, device=trajectory.device)
        else:
            t = s / (total_len + eps)

        return t