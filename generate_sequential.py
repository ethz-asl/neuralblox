import torch
import os
import argparse
from src import config
from src.checkpoints import CheckpointIO
from src import layers
import numpy as np
import open3d as o3d
from tqdm import trange
from os.path import join
from src.common import define_align_matrix, get_shift_noise, get_yaw_noise_matrix
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

pointcloud_sampling = cfg['data'].get("pointcloud_sampling", 0.1)
interval = cfg['test']['scene']['frame_sampling']
noise_shift_std = cfg['test']['scene']['noise']['shift']['shift_std']
shift_on_gravity = cfg['test']['scene']['noise']['shift'].get("on_gravity", False)
noise_yaw_std = cfg['test']['scene']['noise']['yaw_std']
export_pc = cfg['test']['export_pointcloud']

data_path = cfg['data']['path']
intrinsics = cfg['data']['intrinsics']

num_files = len(os.listdir(data_path))
if num_files%2 != 0:
    num_files = num_files - 1

num_frame = int(((len(os.listdir(data_path))) / 2 - 1) * cfg['data']['data_preprocessed_interval'])

if intrinsics == None:
    cam = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
else:
    intrinsic = np.loadtxt(join(data_path, "camera-intrinsics.txt")).tolist()
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = intrinsic

out_dir = cfg['test']['out_dir']
generation_dir = out_dir
mesh_name = cfg['test']['out_name']

# Model
model = config.get_model(cfg, device=device)
model_merging = layers.Conv3D_one_input().to(device)
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io_merging = CheckpointIO(out_dir, model=model_merging)
checkpoint_io.load(join(os.getcwd(), cfg['test']['model_file']))
checkpoint_io_merging.load(join(os.getcwd(),cfg['test']['merging_model_file']))

# Get aligning matrix
align_matrix = define_align_matrix(cfg['data']['align'])
align_matrix = torch.from_numpy(align_matrix).to(device).float()

# get bound
bound_interval = 150
print("Getting scene bounds from dataset sampled every {} frames".format(bound_interval))
sample_points = torch.empty(1, 0, 3, device=device)

for i in trange(0,num_frame, bound_interval):
    # Process data
    depth = join(data_path,"frame-%06d.depth.png"%(i))
    pose = join(data_path,"frame-%06d.pose.txt"%(i))

    pose_np = np.loadtxt(pose)
    depth_raw = o3d.io.read_image(depth)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_raw, cam,depth_trunc=3.0)

    pcl = np.asarray(pcd.points)
    pcl = pcl[::10].T # 3 x N
    N = pcl.shape[1]
    one = np.ones((1, N))
    pcl_h = np.vstack((pcl, one))
    pcl_h = torch.from_numpy(pcl_h).to(device).float()
    pose_np = torch.from_numpy(pose_np).to(device).float()
    c_to_w = torch.matmul(align_matrix, pose_np)
    pcl_h_world = torch.matmul(c_to_w, pcl_h)
    pcl_world = pcl_h_world[:3].T
    sample_points = torch.cat([sample_points, pcl_world.unsqueeze(0)], dim=1)

# Generate
model.eval()
model_merging.eval()
generator = config.get_generator_fusion(model, model_merging, sample_points, cfg, device=device)

if export_pc==True:
    sampled_pcl = np.array([])

num_processed_frame = int((num_frame/interval)+1)

for i in trange(0,num_frame, interval):
    # Process data
    depth = join(data_path, "frame-%06d.depth.png" % (i))
    pose = join(data_path, "frame-%06d.pose.txt" % (i))
    pose_np = np.loadtxt(pose)
    depth_raw = o3d.io.read_image(depth)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_raw, cam, depth_trunc=3.0)
    pcl = np.asarray(pcd.points)

    sampled_index = torch.randperm(len(pcl), device=device)[:int(len(pcl)*pointcloud_sampling)].tolist()
    pcl = pcl[sampled_index].T # 3 x N
    N = pcl.shape[1]
    one = np.ones((1, N))
    pcl_h = np.vstack((pcl, one))
    pcl_h = torch.from_numpy(pcl_h).to(device).float()
    pose_np = torch.from_numpy(pose_np).to(device).float()
    c_to_w = torch.matmul(align_matrix, pose_np)

    if noise_shift_std != 0:
        noise_x, noise_y, noise_z = get_shift_noise(std_x=noise_shift_std, std_y=noise_shift_std,
                                                    std_z=noise_shift_std)
        c_to_w[0 ,3] += noise_x
        c_to_w[2, 3] += noise_z

        if shift_on_gravity == True:
            c_to_w[1, 3] += noise_y

    if noise_yaw_std != 0:
        yaw_matrix = get_yaw_noise_matrix(std=noise_yaw_std, device=device)  # 3x3
        rotation = yaw_matrix
        c_to_w[:3 ,:3] = torch.matmul(rotation, c_to_w[:3 ,:3])

    pcl_h_world = torch.matmul(c_to_w, pcl_h)
    pcl_world = pcl_h_world[:3].T

    if export_pc==True:
        if len(sampled_pcl)==0:
            sampled_pcl = pcl_world.detach().cpu().numpy()
        else:
            sampled_pcl = np.vstack((sampled_pcl, pcl_world.detach().cpu().numpy()))

    input_data = {}
    input_data['inputs'] = pcl_world.unsqueeze(0)

    if i == 0:
        print('\nEncode and fuse latent codes from {} frames'.format((num_processed_frame)))
    latent = generator.generate_latent(input_data)

latent = generator.update_all(latent)
mesh, stats_dict, value_grid = generator.generate_mesh_from_neural_map(latent)

print("Saving mesh")
# Write output
Path(os.path.join(generation_dir, 'mesh')).mkdir(parents= True, exist_ok=True)
mesh_out_file = os.path.join(generation_dir, 'mesh','%s.off' % mesh_name)
mesh.export(mesh_out_file)

if export_pc == True:
    print("Saving sampled point cloud")
    sampled_index = np.random.permutation(len(sampled_pcl))[:2000000]
    sampled_pcl = sampled_pcl[sampled_index]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_pcl)
    Path(os.path.join(generation_dir, 'ply')).mkdir(parents=True, exist_ok=True)
    out_path = join(generation_dir, 'ply','%s.ply' % mesh_name)
    o3d.io.write_point_cloud(out_path, pcd)