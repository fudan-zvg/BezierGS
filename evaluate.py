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
import glob
import json
import os
import torch
import torch.nn.functional as F
from utils.loss_utils import psnr, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight, ColorCorrection, PoseCorrection
from utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf

EPS = 1e-5

@torch.no_grad()
def evaluation(iteration, scene : Scene, renderFunc, renderArgs, env_map=None, color_correction=None, pose_correction=None):
    from lpipsPyTorch import lpips

    scale = scene.resolution_scales[0]

    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                    {'name': 'train', 'cameras': scene.getTrainCameras()})



    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            depth_error = 0.0
            dynamic_psnr = []
            outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
            os.makedirs(outdir, exist_ok=True)
            exp_record_dir = os.path.join(args.model_path, "eval", "exp_record")
            os.makedirs(exp_record_dir, exist_ok=True)
            for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                v, _ = scene.gaussians.get_inst_velocity(viewpoint.timestamp)
                other = [v]

                render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map, color_correction=color_correction, pose_correction=pose_correction, other=other)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                depth = render_pkg['depth']
                alpha = render_pkg['alpha']
                sky_depth = 900
                depth = depth / alpha.clamp_min(EPS)
                feature = render_pkg['feature'] / alpha.clamp_min(EPS)
                v_map = feature
                v_norm_map = v_map.norm(dim=0, keepdim=True)
                v_color = visualize_depth(v_norm_map, near=0.01, far=1)
                if env_map is not None:
                    if args.depth_blend_mode == 0:  # harmonic mean
                        depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                    elif args.depth_blend_mode == 1:
                        depth = alpha * depth + (1 - alpha) * sky_depth
                pts_depth = viewpoint.pts_depth.cuda()
                mask = (pts_depth > 0)
                depth_error += F.l1_loss(depth[mask], pts_depth[mask]).double()

                depth = visualize_depth(depth / scene.scale_factor, near=3, far=200)
                alpha = alpha.repeat(3, 1, 1)
                bbox_mask = viewpoint.bbox_mask.repeat(3, 1, 1)

                gt_dynamic_image = torch.zeros_like(gt_image)
                gt_dynamic_image[bbox_mask] = gt_image[bbox_mask]
                dynamic_render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, color_correction=color_correction, other=other, mask=(scene.gaussians.get_group != 0))
                dynamic_render = dynamic_render_pkg["render"]
                dynamic_alpha = dynamic_render_pkg['alpha']
                dynamic_render = dynamic_render * dynamic_alpha + (1 - dynamic_alpha) * torch.ones_like(dynamic_render)

                static_render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map, color_correction=color_correction, other=other, pose_correction=pose_correction, mask=(scene.gaussians.get_group == 0))
                static_alpha = static_render_pkg['alpha']

                pts_depth_vis = visualize_depth(viewpoint.pts_depth)

                grid = [gt_image, gt_dynamic_image, bbox_mask, pts_depth_vis, 
                        image, dynamic_render / dynamic_alpha.clamp_min(EPS), dynamic_alpha.repeat(3, 1, 1), depth, 
                        v_color, dynamic_render, alpha, static_alpha.repeat(3, 1, 1)]
                grid = make_grid(grid, nrow=4)

                save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))

                frame_id, cam_id = viewpoint.colmap_id // 10, viewpoint.colmap_id % 10
                prefix = f"{frame_id:03d}_{cam_id:01d}_"
                save_image(static_render_pkg["render"], os.path.join(exp_record_dir, prefix + "Background_rgbs.png"))
                save_image(static_alpha, os.path.join(exp_record_dir, prefix + "Background_opacities.png"))
                save_image(depth, os.path.join(exp_record_dir, prefix + "depths.png"))
                save_image(dynamic_alpha, os.path.join(exp_record_dir, prefix + "Dynamic_opacities.png"))
                save_image(dynamic_render, os.path.join(exp_record_dir, prefix + "Dynamic_rgbs.png"))
                save_image(gt_image, os.path.join(exp_record_dir, prefix + "gt_rgbs.png"))
                save_image(image, os.path.join(exp_record_dir, prefix + "rgbs.png"))

                l1_test += F.l1_loss(image, gt_image).double()
                psnr_test += psnr(image, gt_image).double()
                ssim_test += ssim(image, gt_image).double()
                lpips_test += lpips(image, gt_image, net_type='alex').double()
                if bbox_mask.sum() == 0:
                    continue
                dynamic_psnr.append(psnr(image[bbox_mask], gt_image[bbox_mask]).double())
                dynamic_gt_img = gt_image * bbox_mask
                dynamic_gt_img[~bbox_mask] = 1
                save_image(bbox_mask.float(), os.path.join(exp_record_dir, prefix + "bbox_mask.png"))
                save_image(dynamic_gt_img, os.path.join(exp_record_dir, prefix + "dynamic_gt_img.png"))

            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            depth_error /= len(config['cameras'])
            dynamic_psnr = sum(dynamic_psnr) / len(dynamic_psnr)

            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} Dynamic-PSNR {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test, dynamic_psnr))
            with open(os.path.join(args.model_path, "eval", f"metrics_{config['name']}_{iteration}.json"), "a+") as f:
                json.dump({"split": config['name'], "iteration": iteration,
                            "psnr": psnr_test.item(), "ssim": ssim_test.item(), "lpips": lpips_test.item(), "dynamic_psnr": dynamic_psnr.item(), "depth_error": depth_error.item()
                            }, f, indent=1)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default = "configs/base.yaml")
    args, _ = parser.parse_known_args()
    
    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    args.resolution_scales = args.resolution_scales[:1]
    print(args)
    
    seed_everything(args.seed)
    
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    
    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    if args.use_color_correction:
        color_correction = ColorCorrection(args)
        color_correction.training_setup(args)
    else:
        color_correction = None

    if args.use_pose_correction:
        pose_correction = PoseCorrection(args)
        pose_correction.training_setup(args)
    else:
        pose_correction = None

    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args)
    
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint), 
                                    os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
    if color_correction is not None:
        color_correction_checkpoint = os.path.join(os.path.dirname(args.checkpoint), 
                                    os.path.basename(args.checkpoint).replace("chkpnt", "color_correction_chkpnt"))
        (color_correction_params, _) = torch.load(color_correction_checkpoint)
        color_correction.restore(color_correction_params)
    if pose_correction is not None:
        pose_correction_checkpoint = os.path.join(os.path.dirname(args.checkpoint), 
                                    os.path.basename(args.checkpoint).replace("chkpnt", "pose_correction_chkpnt"))
        (pose_correction_params, _) = torch.load(pose_correction_checkpoint)
        pose_correction.restore(pose_correction_params)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    evaluation(first_iter, scene, render, (args, background), env_map=env_map, color_correction=color_correction, pose_correction=pose_correction)

    print("Evaluation complete.")
