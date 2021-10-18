import numpy as np
import subprocess
from pathlib import Path
import os
from os.path import join
import pandas as pd
import time
import argparse

# Arguments
parser = argparse.ArgumentParser(
    description='Evaluate 3D Meshes'
)
parser.add_argument('--demo', action='store_true', help='Trim predicted mesh to match demo GT')
parser.add_argument('--color_mesh', action='store_true',
                    help='Evaluate voxblox mesh (or other meshes with RGB pointcloud info)')
parser.add_argument('--num_pc', type=int, default=200000,
                    help='number of sampled point clouds from mesh')
args = parser.parse_args()

num_pc = args.num_pc
demo = args.demo
color_mesh = args.color_mesh

if demo:
    d = {'model': [],
         'accuracy': [],
         'completeness': [],
         'completeness_o': [],
         'recall_025': [],
         'recall_05': [],
         'recall_075': [],
         'recall_025_o': [],
         'recall_05_o': [],
         'recall_075_o': []
         }
else:
    d = {'model': [],
         'accuracy': [],
         'completeness': [],
         'recall_025': [],
         'recall_05': [],
         'recall_075': [],
         }

mesh_dir = "evaluation"
mesh_gt_dir = join(mesh_dir, "mesh_gt")
mesh_gt = join(mesh_gt_dir, (sorted(os.listdir(mesh_gt_dir))[-1]))
mesh_pred_dir = join(mesh_dir, "mesh_pred")
mesh_pred_list = sorted(os.listdir(mesh_pred_dir))[1:]
print(mesh_pred_list)

eval_dir = "evaluation"
asc_dir = join(eval_dir, "asc")
Path(asc_dir).mkdir(parents=True, exist_ok=True)

for i in range(len(mesh_pred_list)):
    mesh_pred_name = mesh_pred_list[i]
    mesh_pred = join(mesh_pred_dir, mesh_pred_name)

    out_name = mesh_pred_name[:-4] + ".asc"
    out_c_name = "c_" + mesh_pred_name[:-4] + ".asc"
    asc_out = join(asc_dir, out_name)
    asc_c_out = join(asc_dir, out_c_name)
    temp_out = join(eval_dir, "temp.bin")

    command = "CloudCompare -SILENT -AUTO_SAVE OFF -o {} -SAMPLE_MESH POINTS {} -SAVE_CLOUDS FILE {}".format(
        mesh_pred, num_pc, temp_out)
    subprocess.run(command.split())
    time.sleep(1.0)
    command = "CloudCompare -SILENT -AUTO_SAVE OFF -C_EXPORT_FMT ASC -o {} -o {} -c2m_dist -SAVE_CLOUDS FILE {}".format(
        temp_out, mesh_gt, asc_out)
    subprocess.run(command.split())
    command = "CloudCompare -SILENT -AUTO_SAVE OFF -o {} -SAMPLE_MESH POINTS {} -SAVE_CLOUDS FILE {}".format(
        mesh_gt, num_pc, temp_out)
    subprocess.run(command.split())
    time.sleep(1.0)
    command = "CloudCompare -SILENT -AUTO_SAVE OFF -C_EXPORT_FMT ASC -o {} -o {} -c2m_dist -SAVE_CLOUDS FILE {}".format(
        temp_out, mesh_pred, asc_c_out)
    subprocess.run(command.split())

    # Pred to GT
    pred_to_gt = np.loadtxt(asc_out)

    if color_mesh == False:
        pred_to_gt[:, 3] = np.abs(pred_to_gt[:, 3])
    else:
        pred_to_gt[:, 6] = np.abs(pred_to_gt[:, 6])

    if demo:
        trim_x = -1.764147162437439
        trim_z = 0.8207700848579407
        mask_low = pred_to_gt[:, 1] < 0.3
        mask_z = pred_to_gt[:, 2] < trim_z
        mask_excess = np.logical_and(mask_low, mask_z)
        pred_to_gt = pred_to_gt[~mask_excess]
        pred_to_gt = pred_to_gt[pred_to_gt[:, 0] > trim_x]

    # GT to pred
    gt_to_pred = np.loadtxt(asc_c_out)
    gt_to_pred[:, 3] = np.abs(gt_to_pred[:, 3])

    if color_mesh == False:
        mean_dist_acc = np.mean(pred_to_gt[:, 3])
    else:
        mean_dist_acc = np.mean(pred_to_gt[:, 6])

    mean_dist_com = np.mean(gt_to_pred[:, 3])

    recall_025 = np.sum(gt_to_pred[:, 3] < 0.025) / len(gt_to_pred)
    recall_05 = np.sum(gt_to_pred[:, 3] < 0.05) / len(gt_to_pred)
    recall_075 = np.sum(gt_to_pred[:, 3] < 0.075) / len(gt_to_pred)

    if demo:
        # remove ground gt_to_m
        gt_to_pred_no_ground = gt_to_pred[gt_to_pred[:, 1] > 0.15]
        mean_dist_com_no_ground = np.mean(gt_to_pred_no_ground[:, 3])
        recall_no_ground_025 = np.sum(gt_to_pred_no_ground[:, 3] < 0.025) / len(gt_to_pred_no_ground)
        recall_no_ground_05 = np.sum(gt_to_pred_no_ground[:, 3] < 0.05) / len(gt_to_pred_no_ground)
        recall_no_ground_075 = np.sum(gt_to_pred_no_ground[:, 3] < 0.075) / len(gt_to_pred_no_ground)

    d['model'].append(mesh_pred_name[:-4])
    d['accuracy'].append(mean_dist_acc)
    d['completeness'].append(mean_dist_com)

    if demo:
        d['completeness_o'].append(mean_dist_com_no_ground)

    d['recall_025'].append(recall_025)
    d['recall_05'].append(recall_05)
    d['recall_075'].append(recall_075)

    if demo:
        d['recall_025_o'].append(recall_no_ground_025)
        d['recall_05_o'].append(recall_no_ground_05)
        d['recall_075_o'].append(recall_no_ground_075)

df = pd.DataFrame(d)
csv_out = join(eval_dir, "evaluation.csv")
df.to_csv(csv_out, index=False)

command = "rm {}".format(temp_out)
subprocess.run(command.split())
