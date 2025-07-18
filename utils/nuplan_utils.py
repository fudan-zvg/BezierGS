import os
import numpy as np
import cv2
import torch
import json
import open3d as o3d
import math
from glob import glob
from tqdm import tqdm
from utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from utils.general_utils_drivex import matrix_to_quaternion, quaternion_to_matrix_numpy
from plyfile import PlyData, PlyElement
from torchvision.utils import save_image
from collections import defaultdict

waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}

_camera2label = {
    'FRONT': 0,
    'FRONT_LEFT': 1,
    'FRONT_RIGHT': 2,
    'SIDE_LEFT': 3,
    'SIDE_RIGHT': 4,
    'REAR': 5,
    'REAR_LEFT': 6,
    'REAR_RIGHT': 7,
}

_label2camera = {
    0: 'FRONT',
    1: 'FRONT_LEFT',
    2: 'FRONT_RIGHT',
    3: 'SIDE_LEFT',
    4: 'SIDE_RIGHT',
    5: 'REAR',
    6: 'REAR_LEFT',
    7: 'REAR_RIGHT',
}
image_heights = [x//2 for x in [1080 for _ in range(8)]]
image_widths = [x//2 for x in [1920 for _ in range(8)]]
image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])


# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(5):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    for i in range(5):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(5)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4]

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses


# calculate obj pose in world frame
# box_info: box_center_x box_center_y box_center_z box_heading
def make_obj_pose(ego_pose, box_info):
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rotz_matrix
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])

    return obj_pose_vehicle, obj_pose_world


def get_obj_pose_tracking(datadir, selected_frames, ego_poses, cameras=[0, 1, 2, 3, 4]):
    tracklets_ls = []
    objects_info = {}

    tracklet_path = os.path.join(datadir, 'track/track_info.txt')
    tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis.json')

    print(f'Loading from : {tracklet_path}')
    f = open(tracklet_path, 'r')
    tracklets_str = f.read().splitlines()
    tracklets_str = tracklets_str[1:]

    f = open(tracklet_camera_vis_path, 'r')
    tracklet_camera_vis = json.load(f)

    start_frame, end_frame = selected_frames[0], selected_frames[1]

    image_dir = os.path.join(datadir, 'images')
    n_cameras = 5
    n_images = len(os.listdir(image_dir))
    n_frames = n_images // n_cameras
    n_obj_in_frame = np.zeros(n_frames)

    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        object_class = tracklet[2]

        if object_class in ['sign', 'misc']:
            continue

        cameras_vis_list = tracklet_camera_vis[str(track_id)][str(frame_id)]
        join_cameras_list = list(set(cameras) & set(cameras_vis_list))
        if len(join_cameras_list) == 0:
            continue

        if track_id not in objects_info.keys():
            objects_info[track_id] = dict()
            objects_info[track_id]['track_id'] = track_id
            objects_info[track_id]['class'] = object_class
            objects_info[track_id]['class_label'] = waymo_track2label[object_class]
            objects_info[track_id]['height'] = float(tracklet[4])
            objects_info[track_id]['width'] = float(tracklet[5])
            objects_info[track_id]['length'] = float(tracklet[6])
        else:
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], float(tracklet[4]))
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], float(tracklet[5]))
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], float(tracklet[6]))

        tr_array = np.concatenate(
            [np.array(tracklet[:2]).astype(np.float64), np.array([type]), np.array(tracklet[4:]).astype(np.float64)]
        )
        tracklets_ls.append(tr_array)
        n_obj_in_frame[frame_id] += 1

    tracklets_array = np.array(tracklets_ls)
    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())
    num_frames = end_frame - start_frame + 1
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    # Iterate through the tracklets and process object data
    for tracklet in tracklets_array:
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        if start_frame <= frame_id <= end_frame:
            ego_pose = ego_poses[frame_id]
            obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_pose, tracklet[6:10])

            frame_idx = frame_id - start_frame
            obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()

            visible_objects_ids[frame_idx, obj_column] = track_id
            visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle
            visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world

    # Remove static objects
    print("Removing static objects")
    for key in objects_info.copy().keys():
        all_obj_idx = np.where(visible_objects_ids == key)
        if len(all_obj_idx[0]) > 0:
            obj_world_postions = visible_objects_pose_world[all_obj_idx][:, :3]
            distance = np.linalg.norm(obj_world_postions[0] - obj_world_postions[-1])
            dynamic = np.any(np.std(obj_world_postions, axis=0) > 0.5) or distance > 2
            if not dynamic:
                visible_objects_ids[all_obj_idx] = -1.
                visible_objects_pose_vehicle[all_obj_idx] = -1.
                visible_objects_pose_world[all_obj_idx] = -1.
                objects_info.pop(key)
        else:
            objects_info.pop(key)

    # Clip max_num_obj
    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = np.sum(mask, axis=1).max()
    print("Max obj per frame:", max_obj_per_frame_new)

    if max_obj_per_frame_new == 0:
        print("No moving obj in current sequence, make dummy visible objects")
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0
    elif max_obj_per_frame_new < max_obj_per_frame:
        visible_objects_ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        visible_objects_pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        visible_objects_pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for frame_idx in range(num_frames):
            for y in range(max_obj_per_frame):
                obj_id = visible_objects_ids[frame_idx, y]
                if obj_id >= 0:
                    obj_column = np.argwhere(visible_objects_ids_new[frame_idx, :] < 0).min()
                    visible_objects_ids_new[frame_idx, obj_column] = obj_id
                    visible_objects_pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, y]
                    visible_objects_pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, y]

        visible_objects_ids = visible_objects_ids_new
        visible_objects_pose_vehicle = visible_objects_pose_vehicle_new
        visible_objects_pose_world = visible_objects_pose_world_new

    box_scale = 1
    print('box scale: ', box_scale)

    frames = list(range(start_frame, end_frame + 1))
    frames = np.array(frames).astype(np.int32)

    # postprocess object_info
    for key in objects_info.keys():
        obj = objects_info[key]
        if obj['class'] == 'pedestrian':
            obj['deformable'] = True
        else:
            obj['deformable'] = False

        obj['width'] = obj['width'] * box_scale
        obj['length'] = obj['length'] * box_scale

        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
        obj_frame_idx = obj_frame_idx.astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj['start_frame'] = np.min(obj_frames)
        obj['end_frame'] = np.max(obj_frames)

        objects_info[key] = obj

    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz]
    objects_tracklets_world = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_world], axis=-1
    )

    objects_tracklets_vehicle = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1
    )

    return objects_tracklets_world, objects_tracklets_vehicle, objects_info


def padding_tracklets(tracklets, frame_timestamps, min_timestamp, max_timestamp):
    # tracklets: [num_frames, max_obj, ....]
    # frame_timestamps: [num_frames]

    # Clone instead of extrapolation
    if min_timestamp < frame_timestamps[0]:
        tracklets_first = tracklets[0]
        frame_timestamps = np.concatenate([[min_timestamp], frame_timestamps])
        tracklets = np.concatenate([tracklets_first[None], tracklets], axis=0)

    if max_timestamp > frame_timestamps[1]:
        tracklets_last = tracklets[-1]
        frame_timestamps = np.concatenate([frame_timestamps, [max_timestamp]])
        tracklets = np.concatenate([tracklets, tracklets_last[None]], axis=0)

    return tracklets, frame_timestamps


def storePly(path, xyz, rgb):
    # set rgb to 0 - 255
    if rgb.max() <= 1. and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0., 255.)
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def build_pointcloud(args, datadir, object_tracklets_vehicle, object_info, selected_frames, ego_frame_poses, cam_to_worlds, intrinsics, camera_list):
    start_frame, end_frame = selected_frames[0], selected_frames[1]

    print('build point cloud')
    pointcloud_dir = os.path.join(args.model_path, 'input_ply')
    os.makedirs(pointcloud_dir, exist_ok=True)

    points_xyz_dict = dict()
    points_rgb_dict = dict()
    points_xyz_dict['bkgd'] = []
    points_rgb_dict['bkgd'] = []
    for track_id in object_info.keys():
        points_xyz_dict[f'obj_{int(track_id):03d}'] = []
        points_rgb_dict[f'obj_{int(track_id):03d}'] = []
    print('initialize from lidar pointcloud')

    for i, frame in tqdm(enumerate(range(start_frame, end_frame + 1))):
        
        pcd_path = os.path.join(datadir, 'velodyne', f'{frame:02d}.bin')
        lidar = np.fromfile(pcd_path, dtype=np.float32, count=-1).reshape(-1, 6)
        points_xyz_vehicle, intensity, elongation, timestamp_pts = np.split(lidar, [3, 4, 5], axis=1)
        
        # transfrom LiDAR pointcloud from vehicle frame to world frame
        ego_pose = ego_frame_poses[frame]
        points_xyz_vehicle = np.concatenate(
            [points_xyz_vehicle,
             np.ones_like(points_xyz_vehicle[..., :1])], axis=-1
        )
        points_xyz_world = points_xyz_vehicle @ ego_pose.T

        points_rgb = np.zeros_like(points_xyz_world[..., :3])

        for cam in camera_list:
            image_filename = os.path.join(args.source_path, f"image_{cam}", f"{frame:02d}.png")
            image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.
            image = cv2.resize(image, (image_widths[cam], image_heights[cam]))

            c2w = cam_to_worlds[frame * len(camera_list) + cam]
            points_xyz_cam = points_xyz_world @ np.linalg.inv(c2w).T
            # filter out the points behind the camera
            z_mask = points_xyz_cam[:, 2] > 0
            # project points to image plane
            x_n = points_xyz_cam[z_mask, 0] / points_xyz_cam[z_mask, 2]
            y_n = points_xyz_cam[z_mask, 1] / points_xyz_cam[z_mask, 2]
            
            uv_homogeneous = intrinsics[cam] @ np.vstack([x_n, y_n, np.ones_like(x_n)])
            uv = uv_homogeneous[:2, :].T
            uv = np.round(uv).astype(np.int32)
            uv_mask = (uv[:, 0] >= 0) & (uv[:, 0] < image_widths[cam]) & (uv[:, 1] >= 0) & (uv[:, 1] < image_heights[cam])
            uv = uv[uv_mask]

            combined_mask = np.zeros(points_xyz_cam.shape[0], dtype=bool)
            combined_mask[z_mask] = uv_mask
            points_rgb[combined_mask] = image[uv[:, 1], uv[:, 0]]

        # filer points in tracking bbox
        points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)

        for tracklet in object_tracklets_vehicle[i]:
            track_id = int(tracklet[0])
            if track_id >= 0 and str(track_id) in object_info.keys():
                obj_pose_world = np.eye(4)
                obj_pose_world[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                obj_pose_world[:3, 3] = tracklet[1:4]
                world2local = np.linalg.inv(obj_pose_world)

                points_xyz_obj = points_xyz_world @ world2local.T
                points_xyz_obj = points_xyz_obj[..., :3]

                length = object_info[str(track_id)]['length']
                width = object_info[str(track_id)]['width']
                height = object_info[str(track_id)]['height']
                bbox = [[-length / 2, -width / 2, -height / 2], [length / 2, width / 2, height / 2]]
                obj_corners_3d_local = bbox_to_corner3d(bbox)

                points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)
                points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                points_xyz_dict[f'obj_{track_id:03d}'].append(points_xyz_obj[points_xyz_inbbox])
                points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[points_xyz_inbbox])

        points_lidar_xyz = points_xyz_world[~points_xyz_obj_mask][..., :3]

        points_xyz_dict['bkgd'].append(points_lidar_xyz)
        points_rgb_dict['bkgd'].append(points_rgb[~points_xyz_obj_mask])
    initial_num_obj = 20000

    for k, v in points_xyz_dict.items():
        if len(v) == 0:
            continue
        else:
            points_xyz = np.concatenate(v, axis=0)
            points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
            if k == 'bkgd':
                # downsample lidar pointcloud with voxels
                points_lidar = o3d.geometry.PointCloud()
                points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)
                points_xyz_dict['bkgd'] = points_lidar_xyz
                points_rgb_dict['bkgd'] = points_lidar_rgb
            elif k.startswith('obj'):
                if len(points_xyz) > initial_num_obj:
                    random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                    points_xyz = points_xyz[random_indices]
                    points_rgb = points_rgb[random_indices]
                points_xyz_dict[k] = points_xyz
                points_rgb_dict[k] = points_rgb
            else:
                raise NotImplementedError()

    for k in points_xyz_dict.keys():
        points_xyz = points_xyz_dict[k]
        points_rgb = points_rgb_dict[k]
        ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')
        try:
            storePly(ply_path, points_xyz, points_rgb)
            print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
        except:
            print(f'failed to save pointcloud for {k}')

def build_bbox_mask(args, object_tracklets_vehicle, object_info, selected_frames, intrinsics, cam_to_worlds, camera_list):
    start_frame, end_frame = selected_frames[0], selected_frames[1]
    # [frame, cam] -> mask list
    dynamic_mask_info = defaultdict(list)
    bbox_masks = dict()
    for i, frame in tqdm(enumerate(range(start_frame, end_frame + 1))):
        for cam in camera_list:
            view_idx = frame * 10 + cam
            h, w = image_heights[cam], image_widths[cam]
            obj_bound = np.zeros((h, w)).astype(np.uint8)
            obj_tracklets = object_tracklets_vehicle[i]
            ixt, ext = intrinsics[cam], cam_to_worlds[frame * len(camera_list) + cam]
            for obj_tracklet in obj_tracklets:
                track_id = int(obj_tracklet[0])
                if track_id >= 0 and str(track_id) in object_info.keys():
                    obj_pose_world = np.eye(4)
                    obj_pose_world[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
                    obj_pose_world[:3, 3] = obj_tracklet[1:4]
                    obj_length = object_info[str(track_id)]['length']
                    obj_width = object_info[str(track_id)]['width']
                    obj_height = object_info[str(track_id)]['height']
                    bbox = np.array([[-obj_length, -obj_width, -obj_height],
                                     [obj_length, obj_width, obj_height]]) * 0.5
                    corners_local = bbox_to_corner3d(bbox)
                    corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
                    corners_world = corners_local @ obj_pose_world.T  # 3D bounding box in vehicle frame
                    mask = get_bound_2d_mask(
                        corners_3d=corners_world[..., :3],
                        K=ixt,
                        pose=np.linalg.inv(ext),
                        H=h, W=w
                    )
                    obj_bound = np.logical_or(obj_bound, mask)
                    dynamic_mask_info[view_idx].append(mask)
            # save_image(torch.from_numpy(obj_bound).float(), os.path.join(save_dir, f'{frame:06d}_{cam}.png'))
            bbox_masks[view_idx] = obj_bound
    return dynamic_mask_info, bbox_masks
