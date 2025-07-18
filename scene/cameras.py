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
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
import kornia


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx=None, FoVy=None, cx=None, cy=None, fx=None, fy=None, 
                 image=None,
                 image_name=None, uid=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", timestamp=0.0, 
                 resolution=None, image_path=None,
                 pts_depth=None, sky_mask=None, dynamic_mask=None, bbox_mask=None, ncc_scale=1.0
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image = image
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.resolution = resolution
        self.image_path = image_path
        self.ncc_scale = ncc_scale
        self.nearest_id = []
        self.nearest_names = []

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.sky_mask = sky_mask.to(self.data_device) > 0 if sky_mask is not None else sky_mask
        self.dynamic_mask = dynamic_mask.to(self.data_device) > 0 if dynamic_mask is not None else dynamic_mask
        self.bbox_mask = bbox_mask.to(self.data_device) > 0 if bbox_mask is not None else bbox_mask
        self.pts_depth = pts_depth.to(self.data_device) if pts_depth is not None else pts_depth

        self.image_width = resolution[0]
        self.image_height = resolution[1]

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cx is not None:
            self.FoVx = 2 * math.atan(0.5*self.image_width / fx)
            self.FoVy = 2 * math.atan(0.5*self.image_height / fy)
            self.projection_matrix = getProjectionMatrixCenterShift(self.znear, self.zfar, cx, cy, fx, fy,
                                                                    self.image_width, self.image_height).transpose(0, 1).cuda()
        else:
            self.cx = self.image_width / 2
            self.cy = self.image_height / 2
            self.fx = self.image_width / (2 * np.tan(self.FoVx * 0.5))
            self.fy = self.image_height / (2 * np.tan(self.FoVy * 0.5))
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                         fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.timestamp = timestamp
        self.grid = kornia.utils.create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device='cuda')[0]

    def get_world_directions(self, train=False):
        u, v = self.grid.unbind(-1)
        if train:
            directions = torch.stack([(u-self.cx+torch.rand_like(u))/self.fx,
                                        (v-self.cy+torch.rand_like(v))/self.fy,
                                        torch.ones_like(u)], dim=0)
        else:
            directions = torch.stack([(u-self.cx+0.5)/self.fx,
                                        (v-self.cy+0.5)/self.fy,
                                        torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)
        return directions
    
    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.fx/scale, 0, self.cx/scale], [0, self.fy/scale, self.cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    
    def get_image(self):
        original_image = self.original_image # [3, H, W]
        
        # convert the image tensor to grayscale L = R * 299/1000 + G * 587/1000 + B * 114/1000
        # image_gray = (0.299 * original_image[0] + 0.587 * original_image[1] + 0.114 * original_image[2])[None]
        image_gray = original_image[0:1] * 0.299 + original_image[1:2] * 0.587 + original_image[2:3] * 0.114

        return original_image.cuda(), image_gray.cuda()