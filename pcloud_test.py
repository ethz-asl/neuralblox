import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from src.utils.visualize import visualize_pointcloud
from vedo import *

# Initialization
align_matrix = torch.eye(4)

sample_points = torch.empty(0, 3)

data_path = "/home/sam/thesis/matterport3D/test"
depth = os.path.join(data_path, "matterport_depth_images")
poses = os.path.join(data_path, "matterport_camera_poses")
intrinsics = os.path.join(data_path, "matterport_camera_intrinsics")

depth_filelist = [file for file in os.listdir(depth)]
depth_filelist.sort()
curr_point = np.zeros(3)
old_point = np.zeros(3)
ex_old = o3d.io.read_image("/home/sam/thesis/asldoc-2022-MT-Samuel-Neural-Planning/neuralblox/demo/redwood_apartment_13k/frame-000000.depth.png")
for file in tqdm(depth_filelist):
    # Process data
    frame, cam_id = file.split("_d")
    el, yaw = cam_id.split(".png")[0].split("_")
    pose_file = os.path.join(poses, "{}_pose_{}_{}.txt".format(frame, el, yaw))
    intrinsics_file = os.path.join(intrinsics, "{}_intrinsics_{}.txt".format(frame, el))
    depth_file = os.path.join(depth, file)

    pose_np = np.loadtxt(pose_file)
    depth_raw = o3d.io.read_image(depth_file)
    params = np.loadtxt(intrinsics_file)
    cam = o3d.camera.PinholeCameraIntrinsic(int(params[0]), int(params[1]), params[2], params[3], params[4], params[5])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, cam, depth_scale=4000, depth_trunc=3)

    pcl = np.asarray(pcd.points)
    pcl = pcl[::10].T # 3 x N
    N = pcl.shape[1]
    one = np.ones((1, N))
    pcl_h = np.vstack((pcl, one))
    pcl_h = torch.from_numpy(pcl_h).float()
    pose_np = torch.from_numpy(pose_np).float()
    c_to_w = torch.matmul(align_matrix, pose_np)
    pcl_h_world = torch.matmul(c_to_w, pcl_h)
    pcl_world = pcl_h_world[:3].T
    n_tot = pcl_world.shape[0]
    n_subsample = int(n_tot/100)
    subsample = torch.randperm(n_tot)[:n_subsample]
    sample_points = torch.cat([sample_points, pcl_world[subsample]], dim=0)

    curr_point = pose_np[:3, 3].numpy()

# curr_pt = Point(curr_point, r=36, c="red")
pts = Points(sample_points.numpy())
meshpath = "/home/sam/thesis/matterport3D/17DRP5sb8fy/matterport_mesh/bed1a77d92d64f5cbbaaae4feed64ec1/bed1a77d92d64f5cbbaaae4feed64ec1.obj"
mesh = Mesh(meshpath, alpha=0.5, c="blue")
show(pts, mesh, axes=2).close()
