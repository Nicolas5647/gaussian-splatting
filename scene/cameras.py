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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
import torchvision.transforms as T
import scipy
import scipy.interpolate
from PIL import Image

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid, mask,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.resolution = resolution

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.mask = None
        if mask is not None:
            gt_mask = PILtoTorch(mask, resolution)
            self.mask = gt_mask.clamp(0.0, 1.0).to(self.data_device)

class DifixCamera(Camera):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, invdepthmap,
                 image_name, uid, mask,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        self.resolution = resolution
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.depth_params = depth_params
        self.invdepthmap = invdepthmap
        self.mask = mask
        self.trans = trans
        self.scale = scale
        self.data_device = data_device
        self.train_test_exp = train_test_exp
        self.is_test_dataset = is_test_dataset
        self.is_test_view = is_test_view
        self.uid = uid

    def set_image(self, image):
        if isinstance(image, torch.Tensor):
            tensor_cpu = image.cpu()
            to_pil = T.ToPILImage()
            image = to_pil(tensor_cpu)
            
        super().__init__(
            self.resolution, self.colmap_id, self.R, self.T, self.FoVx, self.FoVy, 
            self.depth_params, image, self.invdepthmap, self.image_name, self.uid, self.mask,
            self.trans, self.scale, self.data_device,
            self.train_test_exp, self.is_test_dataset, self.is_test_view
        )
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / (np.linalg.norm(x) + 1e-8)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses (Least Squares)."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def generate_spiral_path(
    poses: np.ndarray,
    bounds: np.ndarray,
    n_frames: int = 120,
    n_rots: int = 2,
    zrate: float = 0.5,
    spiral_scale_f: float = 1.0,
    spiral_scale_r: float = 1.0,
    focus_distance: float = 0.75,
):
    """Calculates a forward facing spiral path for rendering."""
    near_bound = bounds.min()
    far_bound = bounds.max()
    
    # All cameras will point towards the world space point (0, 0, -focal).
    focal = 1 / (((1 - focus_distance) / near_bound + focus_distance / far_bound))
    focal = focal * spiral_scale_f

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = radii * spiral_scale_r
    radii = np.concatenate([radii, [1.0]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses


def generate_ellipse_path_z(
    poses: np.ndarray,
    n_frames: int = 120,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = None,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses (X-Y Plane)."""
    center = focus_point_fn(poses)
    
    if height is None:
        height = poses[:, 2, 3].mean()
        
    offset = np.array([center[0], center[1], height])

    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    low = -sc + offset
    high = sc + offset
    
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    positions = positions[:-1]

    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(center - p, up, p) for p in positions])


def generate_ellipse_path_y(
    poses: np.ndarray,
    n_frames: int = 120,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = None,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses (X-Z Plane)."""
    center = focus_point_fn(poses)
    
    if height is None:
        height = poses[:, 1, 3].mean()
        
    offset = np.array([center[0], height, center[2]])

    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    low = -sc + offset
    high = sc + offset
    
    y_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    y_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                variation
                * (
                    y_low[1]
                    + (y_high - y_low)[1]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
                low[2] + (high - low)[2] * (np.sin(theta) * 0.5 + 0.5),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    positions = positions[:-1]

    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(center - p, up, p) for p in positions])


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
):
    """Creates a smooth spline path between input keyframe camera poses."""
    def poses_to_points(poses, dist):
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return points_to_poses(new_points)


def rotate_all_cameras(poses: np.ndarray, angle_deg: float = 180) -> np.ndarray:
    """Rotates all cameras on their local vertical axis."""
    rad = np.deg2rad(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    R_rot = np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0    ],
        [-sin_a, 0, cos_a]
    ])

    rotated_poses = []
    for pose_orig in poses:
        R_orig = pose_orig[:3, :3]
        t_orig = pose_orig[:3, 3]

        R_new = R_orig @ R_rot

        pose = np.eye(4)
        pose[:3, :3] = R_new
        pose[:3, 3] = t_orig
        rotated_poses.append(pose[:3, :4])

    return np.stack(rotated_poses, axis=0)


import numpy as np


def generate_continuous_path(poses, n_frames, total_rotation_deg, plane, radius):
    center = focus_point_fn(poses)

    if plane == "xy":
        h_axis = 1
        rot_axes = (0, 2)
    else:
        h_axis = 2
        rot_axes = (0, 1)
    positions = np.atleast_2d(poses[:, :3, 3])
    num_poses = len(positions)

    dx = positions[:, rot_axes[0]] - center[rot_axes[0]]
    dy = positions[:, rot_axes[1]] - center[rot_axes[1]]
    thetas = np.unwrap(np.arctan2(dy, dx))
    radii = np.linalg.norm(
        positions[:, rot_axes] - center[None, rot_axes], axis=1
    )
    heights = positions[:, h_axis]

    radii = np.where(radii < 1e-5, 1.0, radii)
    if num_poses > 1:
        rotation_direction = np.sign(thetas[-1] - thetas[0])
        if rotation_direction == 0:
            rotation_direction = 1.0
    else:
        rotation_direction = 1.0

    if num_poses > 2:
        deg_h = min(2, num_poses - 1)
        deg_r = min(1, num_poses - 1)

        p_height = np.polyfit(thetas, heights, deg_h)
        p_radius = np.polyfit(thetas, radii, deg_r)

        fit_h = np.poly1d(p_height)
        fit_r = np.poly1d(p_radius)
    elif num_poses == 2:
        fit_h = np.poly1d(np.polyfit(thetas, heights, 1))
        fit_r = np.poly1d(np.polyfit(thetas, radii, 1))
    else:
        fit_h = lambda t: heights[0]
        fit_r = lambda t: radii[0]

    if radius is not None:
        fit_r = lambda t: radius

    # 4. Plage d'angles
    theta_first = thetas[0]
    rotation_rad = np.deg2rad(total_rotation_deg) * rotation_direction
    theta_end = theta_first + rotation_rad
    theta_start = thetas[-1]

    new_thetas = np.linspace(theta_start, theta_end, n_frames + 1)[1:]

    new_positions = []
    for t in new_thetas:
        r_t = max(fit_r(t), 0.1)
        h_t = fit_h(t)

        pos_x = center[rot_axes[0]] + r_t * np.cos(t)
        pos_z = center[rot_axes[1]] + r_t * np.sin(t)
        pos_y = h_t

        pos = np.zeros(3)
        pos[rot_axes[0]] = pos_x
        pos[rot_axes[1]] = pos_z
        pos[h_axis] = pos_y
        new_positions.append(pos)

    new_positions = np.array(new_positions)

    avg_up = poses[:, :3, 1].mean(axis=0)
    avg_up = avg_up / np.linalg.norm(avg_up)

    return np.stack([viewmatrix(center - p, avg_up, p) for p in new_positions])

def generate_novel_views(cameras, trajectory_type="spiral", n_frames = 120, **kwargs):
    if len(cameras) == 0:
        return []
        
    poses = []
    for cam in cameras:
        w2c_T = cam.world_view_transform.detach().cpu().numpy()
        c2w_T = np.linalg.inv(w2c_T)
        c2w = c2w_T.T
        poses.append(c2w[:3, :4])
    poses = np.stack(poses, axis=0)

    if trajectory_type == "spiral":
        bounds = kwargs.get("bounds", None)
        if bounds is None:
            focus_pt = focus_point_fn(poses)
            dists = np.linalg.norm(poses[:, :3, 3] - focus_pt, axis=-1)
            near = max(1e-3, np.percentile(dists, 10) * 0.5)
            far = max(near + 1e-3, np.percentile(dists, 90) * 1.5)
            bounds = np.array([near, far])
            
        new_poses = generate_spiral_path(
            poses,
            bounds=bounds,
            n_frames=n_frames,
            n_rots=kwargs.get("n_rots", 2),
            zrate=kwargs.get("zrate", 0.5),
            spiral_scale_f=kwargs.get("spiral_scale_f", 1.0),
            spiral_scale_r=kwargs.get("spiral_scale_r", 1.0),
            focus_distance=kwargs.get("focus_distance", 0.75),
        )
        
    elif trajectory_type == "ellipse":
        plane = kwargs.get("plane", "xy")
        variation = kwargs.get("variation", 0.0)
        phase = kwargs.get("phase", 0.0)
        height = kwargs.get("height", None)
        
        if plane == 'xy':
            new_poses = generate_ellipse_path_y(
                poses, 
                n_frames=n_frames, 
                variation=variation, 
                phase=phase, 
                height=height
            )
        elif plane == 'xz':
            new_poses = generate_ellipse_path_z(
                poses, 
                n_frames=n_frames, 
                variation=variation, 
                phase=phase, 
                height=height
            )
        else:
            raise ValueError(f"Plan inconnu : {plane}. Utilisez 'xz' ou 'xy'.")
            
    elif trajectory_type == "interp":
        n_interp = kwargs.get("n_interp", max(1, n_frames // (len(cameras) - 1)))
        spline_degree = kwargs.get("spline_degree", 5)
        smoothness = kwargs.get("smoothness", 0.03)
        rot_weight = kwargs.get("rot_weight", 0.1)
        
        new_poses = generate_interpolated_path(
            poses,
            n_interp=n_interp,
            spline_degree=spline_degree,
            smoothness=smoothness,
            rot_weight=rot_weight
        )
        
    elif trajectory_type == "rotate":
        angle = kwargs.get("angle", 180)
        new_poses = rotate_all_cameras(poses, angle_deg=angle)

    elif trajectory_type == "continuous":
        plane = kwargs.get("plane", "xy")
        total_rotation = kwargs.get("total_rotation",100.0)
        radius = kwargs.get("radius", None)
        
        new_poses = generate_continuous_path(
            poses,
            n_frames=n_frames,
            total_rotation_deg=total_rotation,
            plane=plane,
            radius=radius
        )

    else:
        raise ValueError(f"Type de trajectoire inconnu : {trajectory_type}")

    ref_cam = cameras[0]
    (width, height_res) = ref_cam.resolution
    fovy = ref_cam.FoVy
    fovx = ref_cam.FoVx

    dummy_image = Image.new("RGB", (width, height_res))

    novel_cameras = []
    for i, c2w in enumerate(new_poses):
        c2w_4x4 = np.eye(4)
        c2w_4x4[:3, :4] = c2w
        w2c = np.linalg.inv(c2w_4x4)
        
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        
        cam = DifixCamera(
            resolution=(width, height_res),
            colmap_id=i,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            depth_params=None,
            invdepthmap=None,
            image_name=f"novel_view_{i}",
            uid=i,
            mask=None,
            trans=ref_cam.trans,
            scale=ref_cam.scale,
            data_device="cuda",
            train_test_exp=getattr(ref_cam, 'train_test_exp', False),
            is_test_dataset=getattr(ref_cam, 'is_test_dataset', False),
            is_test_view=getattr(ref_cam, 'is_test_view', False)
        )
        
        cam.set_image(dummy_image)
        
        novel_cameras.append(cam)

    return novel_cameras

def find_nearest_assignments(original_cameras, new_cameras):
    res = []
    for new_cam in new_cameras:
        dist = [(new_cam.camera_center - orig_cam.camera_center).norm().item() for orig_cam in original_cameras]
        idx = np.argmin(dist)
        res.append(original_cameras[idx])
    return res