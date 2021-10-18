import os
import torch
from src.common import (
    add_key, coord2index
)
from src.training import BaseTrainer
import numpy as np

def get_crop_bound(inputs, input_crop_size, query_crop_size):
    ''' Divide a scene into crops, get boundary for each crop

    Args:
        inputs (dict): input point cloud
    '''

    vol_bound = {}

    lb = inputs.min(axis=1).values[0].cpu().numpy() - 0.01
    ub = inputs.max(axis=1).values[0].cpu().numpy() + 0.01
    lb_query = np.mgrid[lb[0]:ub[0]:query_crop_size, \
               lb[1]:ub[1]:query_crop_size, \
               lb[2]:ub[2]:query_crop_size].reshape(3, -1).T
    ub_query = lb_query + query_crop_size
    center = (lb_query + ub_query) / 2
    lb_input = center - input_crop_size / 2
    ub_input = center + input_crop_size / 2
    # number of crops alongside x,y, z axis
    vol_bound['axis_n_crop'] = np.ceil((ub - lb) / query_crop_size).astype(int)
    # total number of crops
    num_crop = np.prod(vol_bound['axis_n_crop'])
    vol_bound['n_crop'] = num_crop
    vol_bound['input_vol'] = np.stack([lb_input, ub_input], axis=1)
    vol_bound['query_vol'] = np.stack([lb_query, ub_query], axis=1)

    return vol_bound

class Trainer(BaseTrainer):
    ''' Trainer object for fusion network.

    Args:
        model (nn.Module): Convolutional Occupancy Network model
        model_merge (nn.Module): fusion network
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples
        query_n (int): number of query points per voxel
        hdim (int): hidden dimension
        depth (int): U-Net depth (3 -> hdim 32 to 128)

    '''

    def __init__(self, model, model_merge, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False, query_n = 8192, unet_hdim = 32, unet_depth = 2):
        self.model = model
        self.model_merge = model_merge
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.max_crop_with_change = None
        self.query_n = query_n
        self.hdim = unet_hdim
        self.factor = 2**unet_depth

        self.reso = None

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_sequence_window(self, data, input_crop_size, query_crop_size, grid_reso, window = 8):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.model_merge.train()
        self.optimizer.zero_grad()

        self.reso = grid_reso
        d = self.reso//self.factor

        device = self.device
        p_in = data.get('inputs').to(device)
        batch_size, T, D = p_in.size()  # seq_length, T, 3
        query_sample_size = self.query_n

        # Get bounding box from sampled all scans in a batch
        sample_points = p_in[:, :, :]
        sample_points = sample_points.view(-1, 3).unsqueeze(0)

        # Shuffle p_in
        p_in =p_in[torch.randperm(p_in.size()[0])]

        vol_bound_all = get_crop_bound(sample_points, input_crop_size, query_crop_size)
        n_crop = vol_bound_all['n_crop']
        n_crop_axis = vol_bound_all['axis_n_crop']

        ## Initialize latent map (prediction)
        latent_map_pred = torch.zeros(n_crop_axis[0], n_crop_axis[1], n_crop_axis[2],
                                      self.hdim*self.factor, d, d, d).to(device)

        loss_all = 0
        crop_with_change_count = None

        counter = 0

        for n in range(batch_size):
            p_input_n = p_in[n].unsqueeze(0)

            # Get prediction
            latent_map_pred, unet, crop_with_change_count = self.update_latent_map_window(p_input_n, latent_map_pred,
                                                                                       n_crop,
                                                                                       vol_bound_all, crop_with_change_count)

            if (n+1)%window==0:
                # Get vol bounds of updated grids
                crop_with_change = list(map(bool, crop_with_change_count))
                vol_bound_valid = []

                for i in range(len(crop_with_change)):
                    if crop_with_change[i]==True:
                        # Get bound of the current crop
                        vol_bound = {}
                        vol_bound['input_vol'] = vol_bound_all['input_vol'][i]

                        vol_bound_valid.append(vol_bound)

                # Merge
                latent_update_pred = self.merge_latent_codes(latent_map_pred, crop_with_change_count)

                # Get prediction logits
                logits_pred, query_points = self.get_logits_and_latent(latent_update_pred, unet, crop_with_change,
                                                                       query_sample_size)

                # Get ground truth
                points_accumulated = p_in[(n+1-window):(n + 1), :, :]
                points_accumulated = points_accumulated.view(-1, 3).unsqueeze(0)
                latent_update_gt, unet_gt = self.update_latent_map_gt(points_accumulated, vol_bound_valid)

                # Get ground truth logits
                crop_with_change = list(map(bool, crop_with_change_count))
                logits_gt, _ = self.get_logits_and_latent(latent_update_gt, unet_gt, crop_with_change,
                                                          query_sample_size, query_points)

                # Calculate loss
                prediction = {}
                gt = {}
                prediction['logits'] = logits_pred
                gt['logits'] = logits_gt

                prediction['latent'] = latent_update_pred
                gt['latent'] = latent_update_gt

                loss = self.compute_sequential_loss(prediction, gt, latent_loss=True)
                loss_all += loss.item()
                counter += 1
                loss.backward()
                self.optimizer.step()

                # Re-initialize
                latent_map_pred = torch.zeros(n_crop_axis[0], n_crop_axis[1], n_crop_axis[2],
                                              self.hdim*self.factor, d, d, d).to(device)

                crop_with_change_count = None

        return loss_all / counter

    def update_latent_map_window(self, p_input, latent_map_pred, n_crop, vol_bound_all, crop_with_change_count=None):
        """
        Sum latent codes and keep track of counts of updated grids
        """

        if crop_with_change_count is None:
            crop_with_change_count = [0] * n_crop

        p_input_mask_list = []
        vol_bound_valid = []
        mask_valid = []

        H, W, D, c, h, w, d = latent_map_pred.size()
        latent_map_pred = latent_map_pred.view(-1, c, h, w, d)

        updated_crop = [False]*n_crop

        for i in range(n_crop):

            # Get bound of the current crop
            vol_bound = {}
            vol_bound['input_vol'] = vol_bound_all['input_vol'][i]

            # Obtain mask
            mask_x = (p_input[:, :, 0] >= vol_bound['input_vol'][0][0]) & \
                       (p_input[:, :, 0] < vol_bound['input_vol'][1][0])
            mask_y = (p_input[:, :, 1] >= vol_bound['input_vol'][0][1]) & \
                       (p_input[:, :, 1] < vol_bound['input_vol'][1][1])
            mask_z = (p_input[:, :, 2] >= vol_bound['input_vol'][0][2]) & \
                       (p_input[:, :, 2] < vol_bound['input_vol'][1][2])
            mask = mask_x & mask_y & mask_z

            p_input_mask = p_input[mask]

            # If first scan is empty in the crop, then continue
            if p_input_mask.shape[0] == 0:  # no points in the current crop
                continue
            else:
                if self.max_crop_with_change is not None:
                    crop_with_change = list(map(bool, crop_with_change_count))
                    if sum(crop_with_change) == self.max_crop_with_change:
                        break

                crop_with_change_count[i] += 1
                p_input_mask_list.append(p_input_mask)
                vol_bound_valid.append(vol_bound)
                mask_valid.append(mask)

                updated_crop[i] = True

        valid_crop_index = np.where(updated_crop)[0].tolist()
        n_crop_update = sum(updated_crop)

        fea = torch.zeros(n_crop_update, self.hdim, self.reso, self.reso, self.reso).to(self.device)

        _, unet = self.encode_crop_sequential(p_input_mask_list[0], self.device, vol_bound=vol_bound_valid[0])
        for i in range(n_crop_update):
            fea[i], _ = self.encode_crop_sequential(p_input_mask_list[i], self.device, vol_bound=vol_bound_valid[i])

        fea, latent_update = unet(fea)
        latent_map_pred[valid_crop_index] += latent_update

        latent_map_pred = latent_map_pred.view(H, W, D, c, h, w, d)

        return latent_map_pred, unet, crop_with_change_count

    def merge_latent_codes(self, latent_map_pred, crop_with_change_count):
        H, W, D, c, h, w, d = latent_map_pred.size()
        latent_map_pred = latent_map_pred.view(-1, c, h, w, d)

        crop_with_change = list(map(bool, crop_with_change_count))

        divisor = torch.FloatTensor(crop_with_change_count)[crop_with_change]
        divisor = divisor.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(self.device)

        latent_update = latent_map_pred[crop_with_change]
        latent_update = torch.div(latent_update, divisor)

        fea_dict = {}
        fea_dict['latent'] = latent_update
        latent_update = self.model_merge(fea_dict)

        return latent_update


    def update_latent_map_gt(self, p_input, vol_bound_valid):

        fea = torch.zeros(len(vol_bound_valid), self.hdim, self.reso, self.reso, self.reso).to(self.device)

        for i in range(len(vol_bound_valid)):
            mask_x = (p_input[:, :, 0] >= vol_bound_valid[i]['input_vol'][0][0]) & \
                     (p_input[:, :, 0] < vol_bound_valid[i]['input_vol'][1][0])
            mask_y = (p_input[:, :, 1] >= vol_bound_valid[i]['input_vol'][0][1]) & \
                     (p_input[:, :, 1] < vol_bound_valid[i]['input_vol'][1][1])
            mask_z = (p_input[:, :, 2] >= vol_bound_valid[i]['input_vol'][0][2]) & \
                     (p_input[:, :, 2] < vol_bound_valid[i]['input_vol'][1][2])
            mask = mask_x & mask_y & mask_z

            fea[i], unet = self.encode_crop_sequential(p_input[mask], self.device,
                                                                        vol_bound=vol_bound_valid[i])

        fea, latent_update = unet(fea)


        return latent_update, unet

    def get_logits_and_latent(self, latent_update, unet, crop_with_change, query_sample_size, query_points=None):

        # Initialize logits list
        num_valid_crops = sum(crop_with_change)

        # Generate query points if not initialized
        if query_points == None:
            sampled_points = torch.rand(num_valid_crops, query_sample_size, 3).to(self.device)
            pi_in = {'p': sampled_points}
            p_n = {}
            p_n['grid'] = sampled_points
            pi_in['p_n'] = p_n
            query_points = pi_in

        # Get latent codes of valid crops
        kwargs = {}
        fea = {}
        fea['unet3d'] = unet
        fea['latent'] = latent_update

        p_r= self.model.decode(query_points, fea, **kwargs)
        logits = p_r.logits

        return logits, query_points

    def encode_crop_sequential(self, inputs, device, fea = 'grid', vol_bound=None):
        ''' Encode a crop to feature volumes

        Args:
            inputs (dict): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        '''

        index = {}
        grid_reso = self.reso
        ind = coord2index(inputs.clone(), vol_bound['input_vol'], reso=grid_reso, plane=fea)
        index[fea] = ind.unsqueeze(0)
        input_cur = add_key(inputs.unsqueeze(0), index, 'points', 'index', device=device)

        fea, unet = self.model.encode_inputs(input_cur)

        return fea, unet

    def compute_sequential_loss(self, prediction, gt, latent_loss = False):

        loss_logits = torch.nn.L1Loss(reduction='mean')
        loss_i = 1 * loss_logits(prediction['logits'], gt['logits'])
        loss = loss_i

        if latent_loss == True:
            loss_latent = torch.nn.L1Loss(reduction='mean')
            loss_ii = 1 * loss_latent(prediction['latent'], gt['latent'])
            loss += loss_ii

        return loss

