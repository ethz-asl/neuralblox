import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
from src.training import BaseTrainer
from math import sin,cos,radians,sqrt
import random

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, DEGREES = 0):
        ''' Performs a training step.
        Args:
            data (dict): data dictionary
            DEGREES (integer): degree range in which object is going to be rotated
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, DEGREES = DEGREES)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)
        
        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, 
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def compute_loss(self, data, DEGREES = 0):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        if (DEGREES != 0):
            inputs, rotation = self.rotate_points(inputs, DEGREES=DEGREES)
            p = self.rotate_points(p, use_rotation_tensor=True)
        
        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        c = self.model.encode_inputs(inputs)

        kwargs = {}
        # General points
        logits = self.model.decode(p, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss_i.sum(-1).mean()

        return loss

    def rotate_points(self, pointcloud_model, DEGREES=0, query_points=False, use_rotation_tensor=False,
                      save_rotation_tensor=False):
        ## https://en.wikipedia.org/wiki/Rotation_matrix
        """
            Function for rotating points
            Args:
                pointcloud_model (numpy 3d array) - batch_size x pointcloud_size x 3d channel sized numpy array which presents pointcloud
                DEGREES (int) - range of rotations to be used
                query_points (boolean) - used for rotating query points with already existing rotation matrix
                use_rotation_tensor (boolean) - asking whether DEGREES should be used for generating new rotation matrix, or use the already established one
                save_rotation_tensor (boolean) - asking to keep rotation matrix in a pytorch .pt file
        """
        if (use_rotation_tensor != True):
            angle_range = DEGREES
            x_angle = radians(random.uniform(-1,1) * 5)
            y_angle = radians(random.uniform(-1,1) * 180)
            z_angle = radians(random.uniform(-1,1) * 5)

            rot_x = torch.Tensor(
                [[1, 0, 0, 0], [0, cos(x_angle), -sin(x_angle), 0], [0, sin(x_angle), cos(x_angle), 0], [0, 0, 0, 1]])
            rot_y = torch.Tensor(
                [[cos(y_angle), 0, sin(y_angle), 0], [0, 1, 0, 0], [-sin(y_angle), 0, cos(y_angle), 0], [0, 0, 0, 1]])
            rot_z = torch.Tensor(
                [[cos(z_angle), -sin(z_angle), 0, 0], [sin(z_angle), cos(z_angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            rotation_matrix = torch.mm(rot_z, rot_y)
            rotation_matrix = torch.mm(rot_x, rotation_matrix)
            rotation_matrix = torch.transpose(rotation_matrix, 0, 1)

            batch_size, point_cloud_size, _ = pointcloud_model.shape
            pointcloud_model = torch.cat(
                [pointcloud_model, torch.ones(batch_size, point_cloud_size, 1).to(self.device)], dim=2)

            pointcloud_model_rotated = torch.matmul(pointcloud_model, rotation_matrix.to(self.device))
            self.rotation_matrix = rotation_matrix

            if (save_rotation_tensor):
                torch.save(rotation_matrix, 'rotation_matrix.pt')  # used for plane prediction, change it at your will
            return pointcloud_model_rotated[:, :, 0:3], (x_angle, y_angle, z_angle)
        else:
            batch_size, point_cloud_size, _ = pointcloud_model.shape
            #pointcloud_model = pointcloud_model / sqrt(0.55 ** 2 + 0.55 ** 2 + 0.55 ** 2)
            pointcloud_model = torch.cat(
                [pointcloud_model, torch.ones(batch_size, point_cloud_size, 1).to(self.device)], dim=2)
            pointcloud_model_rotated = torch.matmul(pointcloud_model, self.rotation_matrix.to(self.device))
            return pointcloud_model_rotated[:, :, 0:3]

