# Description: Load the EmerWaymo dataset for training and testing

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud, focal2fov
from utils.general_utils import load_ply
from utils.nuplan_utils import build_pointcloud, build_bbox_mask
from utils.general_utils_drivex import quaternion_to_matrix_numpy
from collections import defaultdict
import cv2
import json

image_filename_to_cam = lambda x: int(x.split('.')[0][-1])

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius > 0:
        scale_factor = 1. / fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor

def readnuPlanInfo(args):
    load_bbox_mask = args.load_bbox_mask
    load_dynamic_mask = args.load_dynamic_mask
    neg_fov = args.neg_fov
    start_time = args.start_time
    end_time = args.end_time

    ORIGINAL_SIZE = [[1080, 1920] for _ in range(8)]
    load_size = [540, 960]

    num_seqs = len(os.listdir(os.path.join(args.source_path, "velodyne")))
    if end_time == -1:
        end_time = int(num_seqs)
    else:
        end_time += 1

    frame_num = end_time - start_time
    time_duration = args.time_duration
    time_interval = (time_duration[1] - time_duration[0]) / (end_time - start_time)
    
    cam_infos = []
    points = []
    points_time = []

    data_root = args.source_path

    camera_list = args.camera_list
    truncated_min_range, truncated_max_range = 2, 80

    # ---------------------------------------------
    # load poses: intrinsic, c2w, l2w per camera
    # ---------------------------------------------
    _intrinsics = []
    cam_to_egos = []
    for i in camera_list:
        # load intrinsics
        intrinsic = np.loadtxt(os.path.join(data_root, "calib", f"intrinsic.txt"))
        fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
        fx, fy = (
            fx * load_size[1] / ORIGINAL_SIZE[i][1],
            fy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        cx, cy = (
            cx * load_size[1] / ORIGINAL_SIZE[i][1],
            cy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        _intrinsics.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

        # load extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_root, "calib", f"cam{i}_2_lidar.txt"))
        cam_to_egos.append(cam_to_ego)

    ego_pose_dir = os.path.join(data_root, "pose")
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(len(camera_list))]
    for ego_pose_path in ego_pose_paths:
        ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
        ego_frame_poses.append(ego_frame_pose)

        for i in range(len(camera_list)):
            ego_cam_poses[i].append(ego_frame_pose)
    
    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses) # [num_frames, 4, 4], lidar/ego -> world

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(len(camera_list))]
    ego_cam_poses = np.array(ego_cam_poses) # [num_cameras, num_frames, 4, 4], lidar/ego -> world

    # ---------------------------------------------
    # get c2w and w2c transformation per frame and camera 
    # ---------------------------------------------
    # compute per-image poses and intrinsics
    cam_to_worlds, ego_to_worlds = [], []
    lidar_to_worlds = []
    for t in range(start_time, end_time):
        # ego to world transformation: cur_ego -> world
        ego_to_world = ego_frame_poses[t]
        ego_to_worlds.append(ego_to_world)
        for cam_id in camera_list:
            # transformation:
            # opencv_cam -> waymo_cam -> waymo_cur_ego -> world
            cam2world = ego_cam_poses[cam_id, t] @ cam_to_egos[cam_id]
            cam_to_worlds.append(cam2world)
        # lidar to world : lidar = ego in nuPlan
        lidar_to_worlds.append(ego_to_world)  # same as ego_to_worlds
    # convert to numpy arrays
    cam_to_worlds = np.stack(cam_to_worlds, axis=0)
    lidar_to_worlds = np.stack(lidar_to_worlds, axis=0)

    # len(object_info) = num_objects
    obj_info_path = os.path.join(data_root, "track", "object_info.json")
    with open(obj_info_path, 'r') as f:
        object_info = json.load(f)
    # object_tracklets_vehicle.shape = [num_frames, num_objects, 8], 8: id(1) + position(3) + quaternion(4)
    obj_tracklets_path = os.path.join(data_root, "track", "object_tracklets_vehicle.npy")
    object_tracklets_vehicle = np.load(obj_tracklets_path)
    build_pointcloud(args, data_root, object_tracklets_vehicle, object_info,
                         [start_time, end_time - 1], ego_frame_poses, cam_to_worlds, _intrinsics, camera_list)

    os.makedirs(os.path.join(args.source_path, "dynamic_mask_select"), exist_ok=True)
    dynamic_bbox_info, bbox_mask_list = build_bbox_mask(args, object_tracklets_vehicle, object_info,
                        [start_time, end_time - 1], _intrinsics, cam_to_worlds, camera_list)

    # ---------------------------------------------
    # get image, sky_mask, lidar per frame and camera 
    # ---------------------------------------------
    os.makedirs(os.path.join(args.source_path, "refined_bbox"), exist_ok=True)
    for idx, t in enumerate(tqdm(range(start_time, end_time), desc="Loading data", bar_format='{l_bar}{bar:50}{r_bar}')):
        images = []
        image_paths = []
        HWs = []
        sky_masks = []
        dynamic_masks = []
        bbox_masks = []

        for cam_idx in camera_list:
            image_path = os.path.join(args.source_path, f"image_{cam_idx}", f"{t:02d}.png")
            im_data = Image.open(image_path)
            im_data = im_data.resize((load_size[1], load_size[0]), Image.BILINEAR)  # PIL resize: (W, H)
            W, H = im_data.size
            image = np.array(im_data) / 255.
            HWs.append((H, W))
            images.append(image)
            image_paths.append(image_path)

            sky_path = os.path.join(args.source_path, f'mask_{cam_idx}', f"{t:02d}_sky.png")
            sky_data = Image.open(sky_path)
            sky_data = sky_data.resize((load_size[1], load_size[0]), Image.NEAREST)  # PIL resize: (W, H)
            sky_mask = np.array(sky_data) > 0
            sky_masks.append(sky_mask.astype(np.float32))

            if load_bbox_mask:
                view_idx = t * 10 + cam_idx
                bbox_mask = bbox_mask_list[view_idx]
                bbox_masks.append(bbox_mask.astype(np.float32))

            if load_dynamic_mask:
                view_idx = t * 10 + cam_idx
                dynamic_bbox_list = dynamic_bbox_info[view_idx]

                seg_path = os.path.join(args.source_path, "seg_npy", f"mask_{cam_idx}", f"{t:02d}.npy")
                seg_data = np.load(seg_path)
                # resize
                seg_data = cv2.resize(seg_data, (load_size[1], load_size[0]), interpolation=cv2.INTER_NEAREST)
                max_seg_value = seg_data.max()
                bbox_dict = defaultdict(list)
                for seg_value in range(1, max_seg_value + 1):
                    seg_mask = (seg_data == seg_value)
                    if seg_mask.sum() == 0:
                        continue
                    ideal_bbox = -1
                    max_overlap = 0
                    for bbox_idx, dynamic_bbox in enumerate(dynamic_bbox_list):
                        overlap = dynamic_bbox[seg_mask].sum()
                        if overlap > 0.5 * seg_mask.sum() and overlap > max_overlap:
                            max_overlap = overlap
                            ideal_bbox = bbox_idx
                    if ideal_bbox != -1:
                        bbox_dict[ideal_bbox].append(seg_mask)

                cur_view_mask = np.zeros_like(seg_data)
                for bbox_idx, seg_masks in bbox_dict.items():
                    cur_bbox_mask = max(seg_masks, key=lambda x: dynamic_bbox_list[bbox_idx][x].sum() / np.logical_or(dynamic_bbox_list[bbox_idx], x).sum()) # IoU
                    cur_view_mask = np.logical_or(cur_view_mask, cur_bbox_mask)

                if load_size == [1080, 1920]:
                    refined_bbox_path = os.path.join(args.source_path, "refined_bbox", f"{t:06d}_{cam_idx}.png")
                    Image.fromarray(cur_view_mask.astype(np.uint8) * 255).save(refined_bbox_path)

                dynamic_masks.append(cur_view_mask.astype(np.float32))

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (frame_num - 1)

        lidar_info = np.fromfile(os.path.join(args.source_path, "velodyne", f"{t:02d}.bin"), dtype=np.float32, count=-1).reshape(-1, 6)
        lidar_points, intensity, elongation, timestamp_pts = np.split(lidar_info, [3, 4, 5], axis=1)
        lidar_points = lidar_points
        # transform lidar points to world coordinate system
        lidar_points = (
                lidar_to_worlds[idx][:3, :3] @ lidar_points.T
                + lidar_to_worlds[idx][:3, 3:4]
        ).T  # point_xyz_world

        points.append(lidar_points)
        point_time = np.full_like(lidar_points[:, :1], timestamp)
        points_time.append(point_time)

        for cam_idx in camera_list:
            # world-lidar-pts --> camera-pts : w2c
            c2w = cam_to_worlds[int(len(camera_list)) * idx + cam_idx]
            w2c = np.linalg.inv(c2w)
            point_camera = (
                    w2c[:3, :3] @ lidar_points.T
                    + w2c[:3, 3:4]
            ).T

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            K = _intrinsics[cam_idx]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            height, width = HWs[cam_idx]
            if neg_fov:
                FovY = -1.0
                FovX = -1.0
            else:
                FovY = focal2fov(fy, height)
                FovX = focal2fov(fx, width)

            image = images[cam_idx]
            sky_mask = sky_masks[cam_idx]
            dynamic_mask = dynamic_masks[cam_idx] if load_dynamic_mask else None
            bbox_mask = bbox_masks[cam_idx] if load_bbox_mask else None
            cam_infos.append(CameraInfo(uid=idx * 10 + cam_idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image=image,
                                        image_path=image_paths[cam_idx], image_name=f"{t:03d}_{cam_idx}",
                                        width=width, height=height, timestamp=timestamp,
                                        pointcloud_camera=point_camera,
                                        fx=fx, fy=fy, cx=cx, cy=cy,
                                        sky_mask=sky_mask,
                                        dynamic_mask=dynamic_mask,
                                        bbox_mask=bbox_mask))

    pointcloud = np.concatenate(points, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform_pca, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)
    # if args.static_thresh > 0: # for PVG separation
    #     args.static_thresh = float(args.static_thresh * scale_factor)

    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data", bar_format='{l_bar}{bar:50}{r_bar}')):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform_pca.T)[:, :3]
    if args.eval:
        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]

        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num) > 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1 / nerf_normalization['radius']

    pcd = None

    # stage1: read point3d.ply, and initialize as static gaussians
    ply_path = os.path.join(args.model_path, "points3d.ply")
    if not os.path.exists(ply_path):
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0], 3]), normals=None, time=pointcloud_timestamp)

    obj2world_dict = dict()
    obj_timestamp_list = dict()
    for track_id in object_info.keys():
        track_id = int(track_id)
        obj2world_dict[f'obj_{track_id:03d}'] = []
        obj_timestamp_list[f'obj_{track_id:03d}'] = []

    for i, frame in tqdm(enumerate(range(start_time, end_time))):
        # if args.eval:
        #     if (i + 1) % args.testhold == 0:
        #         continue
        for tracklet in object_tracklets_vehicle[i]:
            track_id = int(tracklet[0])
            if track_id >= 0 and str(track_id) in object_info.keys():
                obj_pose_world = np.eye(4)
                obj_pose_world[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                obj_pose_world[:3, 3] = tracklet[1:4]  # object -> world
                obj2world_dict[f'obj_{track_id:03d}'].append(obj_pose_world)
                timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * i / (frame_num - 1)
                obj_timestamp_list[f'obj_{track_id:03d}'].append(timestamp)

    ply_dict = dict()
    ply_dict['bkgd'] = {'xyz_array': None, 'colors_array': None, 'start_frame': start_time, 'end_frame': end_time - 1}
    for k, v in object_info.items():
        k = int(k)
        ply_dict[f'obj_{k:03d}'] = {'xyz_offset': None, 'trajectory': None, 'colors_array': None, 'start_frame': v['start_frame'], 'end_frame': v['end_frame'], 'timestamp_list': obj_timestamp_list[f'obj_{k:03d}']}

    for idx, item in enumerate(sorted(os.listdir(os.path.join(args.model_path, "input_ply")))):
        # 0 is background
        if idx == 0:
            xyz, colors = load_ply(os.path.join(args.model_path, "input_ply", item))
            xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
            transform = transform_pca
            xyz = xyz @ transform.T
            ply_dict['bkgd']['xyz_array'] = xyz[:, :3]
            ply_dict['bkgd']['colors_array'] = colors
            continue

        obj_idx = int(item.split('_')[2].split('.')[0])
        xyz, colors = load_ply(os.path.join(args.model_path, "input_ply", item))

        cur_obj_offset_list = []
        cur_obj_trajectory = []
        xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
        ply_dict[f'obj_{obj_idx:03d}']['colors_array'] = colors
        for obj2world in obj2world_dict[f'obj_{obj_idx:03d}']:
            # object -> cur ego -> world
            transform = obj2world
            transform = transform_pca @ transform
            xyz_world = xyz @ transform.T
            xyz_world = xyz_world[:, :3]
            # xyz offset(from trajectory)
            trajectory_pos = transform[:3, 3]
            xyz_offset = xyz_world - trajectory_pos
            cur_obj_offset_list.append(xyz_offset)
            cur_obj_trajectory.append(trajectory_pos)
        ply_dict[f'obj_{obj_idx:03d}']['xyz_offset'] = np.stack(cur_obj_offset_list, axis=1) # [N, T, 3]
        ply_dict[f'obj_{obj_idx:03d}']['trajectory'] = np.stack(cur_obj_trajectory, axis=0) # [N, T, 3]

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           ply_dict=ply_dict,
                           time_interval=time_interval,
                           time_duration=time_duration,
                           scale_factor=scale_factor)

    return scene_info