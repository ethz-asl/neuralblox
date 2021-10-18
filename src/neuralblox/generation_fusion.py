import torch
import numpy as np
from tqdm import trange
import trimesh
from src.utils import libmcubes
from src.common import normalize_coord, add_key, coord2index
import time

class Generator3D(object):
    '''  Generator class for scene generation and latent code fusion.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        model_merge (nn.Module): trained fusion network
        sample points (torch.Tensor): sampled points to define maximum boundaries of scene
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value to define {occupied, free} and for mesh generation
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): query point density (resolution0 points/voxel)
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
        voxel_threshold (float): threshold to define whether or not to encode points to latent code
        boundary_interpolation (bool): whether boundary interpolation is performed
    '''

    def __init__(self, model, model_merge, sample_points, points_batch_size=3000000,
                 threshold=0.05, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 padding=0.1,
                 input_type=None,
                 vol_info=None,
                 vol_bound=None,
                 voxel_threshold = 0.01,
                 boundary_interpolation=False,
                 unet_hdim = 32,
                 unet_depth = 2):
        self.model = model.to(device)
        self.model_merge = model_merge.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.input_type = input_type
        self.padding = padding

        # Merge all scans
        self.latent = None
        self.pointcloud_all = sample_points
        self.unet = None
        self.grid_reso = None
        self.voxel_threshold = voxel_threshold
        self.init_get_bound = None
        self.n_crop = None
        self.n_crop_axis = None
        self.crop_with_change_count = None
        self.bound_interpolation = boundary_interpolation
        self.unet_hdim = unet_hdim
        self.unet_depth = unet_depth

        # for pointcloud_crop
        self.vol_bound = vol_bound
        self.grid_reso = vol_bound['reso']

        if vol_info is not None:
            self.input_vol, _, _ = vol_info

    def generate_latent(self, data):
        ''' Generates voxels of latent codes from input point clouds.
            Adapt for real-world scale.

        Args:
            data (tensor): data tensor
        '''
        inputs = data.get('inputs', torch.empty(1, 0)).to(self.device)
        threshold = self.voxel_threshold * inputs.size()[1]

        # acquire the boundary for every crops
        if self.init_get_bound == None:
            self.get_crop_bound(self.pointcloud_all)
            self.n_crop = self.vol_bound['n_crop']
            self.n_crop_axis = self.vol_bound['axis_n_crop']
            self.grid_reso = self.vol_bound['reso']
            self.init_get_bound = True

        factor = 2**self.unet_depth
        d = int(self.grid_reso / factor)
        if self.latent == None:
            self.latent = torch.zeros(self.n_crop_axis[0] * self.n_crop_axis[1] * self.n_crop_axis[2],
                                      self.unet_hdim*factor, d, d, d).to(self.device)

        if self.crop_with_change_count is None:
            self.crop_with_change_count = [0] * self.n_crop

        crop_with_change = [False] * self.n_crop

        p_input_valid = []

        crop_list = self.get_to_be_processed_crops(inputs)
        n_crop = len(crop_list)

        if n_crop == 0:
            print("WARNING: no valid crop")
            return self.latent

        if self.device =='cpu':
            for i in crop_list:
                vol_bound = {}
                vol_bound['query_vol'] = self.vol_bound['query_vol'][i]
                vol_bound['input_vol'] = self.vol_bound['input_vol'][i]

                # Check if crop is not empty
                mask_x = (inputs[:, :, 0] >= vol_bound['input_vol'][0][0]) & \
                         (inputs[:, :, 0] < vol_bound['input_vol'][1][0])
                mask_y = (inputs[:, :, 1] >= vol_bound['input_vol'][0][1]) & \
                         (inputs[:, :, 1] < vol_bound['input_vol'][1][1])
                mask_z = (inputs[:, :, 2] >= vol_bound['input_vol'][0][2]) & \
                         (inputs[:, :, 2] < vol_bound['input_vol'][1][2])
                mask = mask_x & mask_y & mask_z

                p_input = inputs[mask]

                if p_input.shape[0] == 0:  # no points in the current crop
                    continue

                if threshold is not None:
                    if p_input.shape[0] > threshold:
                        crop_with_change[i] = True
                        p_input_valid.append(p_input)
                        self.crop_with_change_count[i] += 1
                else:
                    crop_with_change[i] = True
                    p_input_valid.append(p_input)
                    self.crop_with_change_count[i] += 1

        else:
            max_processed_crop = int(500*25000/inputs.size(1)) # Fit in 4 GB GPU
            crop_batch_start_end = []

            a = 0
            while a < n_crop:
                if a+max_processed_crop < n_crop:
                    crop_batch_start_end.append(range(a, a+max_processed_crop))
                else:
                    crop_batch_start_end.append(range(a, n_crop))
                a += max_processed_crop

            for b in range(len(crop_batch_start_end)):
                batch_range = crop_batch_start_end[b]
                n_crop_batch = len(batch_range)

                inputs_all = torch.cat([inputs] * n_crop_batch, dim=0)
                vol_bound_all = torch.from_numpy(self.vol_bound['input_vol']).to(self.device)
                vol_bound_all = vol_bound_all.reshape(self.n_crop, -1)[crop_list[batch_range[0]:(batch_range[-1]+1)]]  # n_crop * 6

                # 0: LB x, 3: UB x     1: LB y, 4: UB y      2: LB z, 5 UB z

                # Filter x all
                inputs_all[:, :, 0][inputs_all[:, :, 0] < vol_bound_all[:, 0].unsqueeze(1)] = 0.0
                inputs_all[:, :, 0][inputs_all[:, :, 0] > vol_bound_all[:, 3].unsqueeze(1)] = 0.0

                # Filter y all
                inputs_all[:, :, 1][inputs_all[:, :, 1] < vol_bound_all[:, 1].unsqueeze(1)] = 0.0
                inputs_all[:, :, 1][inputs_all[:, :, 1] > vol_bound_all[:, 4].unsqueeze(1)] = 0.0

                # Filter z all
                inputs_all[:, :, 2][inputs_all[:, :, 2] < vol_bound_all[:, 2].unsqueeze(1)] = 0.0
                inputs_all[:, :, 2][inputs_all[:, :, 2] > vol_bound_all[:, 5].unsqueeze(1)] = 0.0

                mask_all = torch.sum((inputs_all != 0), dim=2) == 3

                for i in range(n_crop_batch):

                    p_input = inputs_all[i][mask_all[i]]

                    if p_input.shape[0] == 0:  # no points in the current crop
                        continue

                    if threshold is not None:
                        if p_input.shape[0] > threshold:
                            changed_index = crop_list[i + batch_range[0]]
                            crop_with_change[changed_index] = True
                            p_input_valid.append(p_input)
                            self.crop_with_change_count[changed_index] += 1
                    else:
                        changed_index = crop_list[i + batch_range[0]]
                        crop_with_change[changed_index] = True
                        p_input_valid.append(p_input)
                        self.crop_with_change_count[changed_index] += 1

        vol_bound_valid_input = self.vol_bound['input_vol'][crop_with_change]
        vol_bound_valid_query = self.vol_bound['query_vol'][crop_with_change]

        if sum(crop_with_change) == 0:
            print("WARNING: No pointcloud inside query volume")
            return self.latent

        fea_grid = torch.zeros(sum(crop_with_change), self.unet_hdim, self.grid_reso,
                               self.grid_reso, self.grid_reso, device=self.device)

        for i in range(sum(crop_with_change)):
            vol_bound = {}
            vol_bound['query_vol'] = vol_bound_valid_query[i]
            vol_bound['input_vol'] = vol_bound_valid_input[i]

            fea_grid[i], self.unet = self.encode_crop_sequential(p_input_valid[i], self.device, vol_bound=vol_bound)

        with torch.no_grad():
            latent = self.unet.encoders[0].basic_module(fea_grid)
            latent = self.unet.encoders[1].pooling(latent)
            latent = self.unet.encoders[1].basic_module(latent)
            latent = self.unet.encoders[2].pooling(latent)
            latent = self.unet.encoders[2].basic_module(latent)

            self.latent[crop_with_change] += latent

        return self.latent

    def get_to_be_processed_crops(self, inputs):
        ''' Obtain voxel indexes to be encoded

        Args:
            inputs (torch.Tensor): input point cloud
        '''

        input_min_x, input_max_x = torch.min(inputs[0][:, 0]), torch.max(inputs[0][:, 0])
        input_min_y, input_max_y = torch.min(inputs[0][:, 1]), torch.max(inputs[0][:, 1])
        input_min_z, input_max_z = torch.min(inputs[0][:, 2]), torch.max(inputs[0][:, 2])

        index_min_x = int((input_min_x - self.vol_bound['query_vol'][0][0][0]) / self.vol_bound['query_crop_size'])
        index_min_y = int((input_min_y - self.vol_bound['query_vol'][0][0][1]) / self.vol_bound['query_crop_size'])
        index_min_z = int((input_min_z - self.vol_bound['query_vol'][0][0][2]) / self.vol_bound['query_crop_size'])
        index_max_x = int((input_max_x - self.vol_bound['query_vol'][0][0][0]) / self.vol_bound['query_crop_size'])
        index_max_y = int((input_max_y - self.vol_bound['query_vol'][0][0][1]) / self.vol_bound['query_crop_size'])
        index_max_z = int((input_max_z - self.vol_bound['query_vol'][0][0][2]) / self.vol_bound['query_crop_size'])

        if index_min_x < 0:
            index_min_x = 0

        if index_min_y < 0:
            index_min_y = 0

        if index_min_z < 0:
            index_min_z = 0

        if index_max_x > (self.n_crop_axis[0] - 1):
            index_max_x = (self.n_crop_axis[0] - 1)

        if index_max_y > (self.n_crop_axis[1] - 1):
            index_max_y = (self.n_crop_axis[1] - 1)

        if index_max_z > (self.n_crop_axis[2] - 1):
            index_max_z = (self.n_crop_axis[2] - 1)

        i_min = index_min_x * (self.n_crop_axis[1] * self.n_crop_axis[2]) + index_min_y * self.n_crop_axis[
            2] + index_min_z
        i_max = index_max_x * (self.n_crop_axis[1] * self.n_crop_axis[2]) + index_max_y * self.n_crop_axis[
            2] + index_max_z

        # remove unneccesary index
        crop_np = np.array(list(range(i_min, i_max + 1)))
        remainder_x_np = np.remainder(crop_np, (self.n_crop_axis[1] * self.n_crop_axis[2]))
        index_y_np = np.floor_divide(remainder_x_np, self.n_crop_axis[2])
        mask_y_np = index_y_np <= index_max_y
        index_z_np = np.remainder(remainder_x_np, self.n_crop_axis[2])
        mask_z_np = index_z_np <= index_max_z
        mask_yz = np.logical_and(mask_y_np, mask_z_np).tolist()
        crop_list = crop_np[mask_yz].tolist()

        return crop_list

    def update_all(self, latent):
        ''' Perform latent code fusion for all voxels

        Args:
            latent (torch.Tensor): summed latent codes in a map
        '''

        factor = 2**self.unet_depth
        d = int(self.grid_reso/factor)

        for i in range(self.n_crop):
            if self.crop_with_change_count[i] != 0:
                fea_dict = {}
                fea_dict['latent'] = latent[i].unsqueeze(0) / self.crop_with_change_count[i]
                latent[i] = self.model_merge(fea_dict)

        return latent.view(self.n_crop_axis[0], self.n_crop_axis[1], self.n_crop_axis[2], self.unet_hdim*factor, d, d, d)

    def get_crop_bound(self, inputs):
        ''' Divide a scene into crops, get boundary for each crop

        Args:
            inputs (torch.Tensor): input point cloud
        '''
        query_crop_size = self.vol_bound['query_crop_size']
        input_crop_size = self.vol_bound['input_crop_size']

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
        self.vol_bound['axis_n_crop'] = np.ceil((ub - lb) / query_crop_size).astype(int)
        # total number of crops
        num_crop = np.prod(self.vol_bound['axis_n_crop'])
        self.vol_bound['n_crop'] = num_crop
        self.vol_bound['input_vol'] = np.stack([lb_input, ub_input], axis=1)
        self.vol_bound['query_vol'] = np.stack([lb_query, ub_query], axis=1)

    def encode_crop_sequential(self, inputs, device, vol_bound=None):
        ''' Encode a crop to feature volumes (before U-Net)

        Args:
            inputs (dict): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        '''
        if vol_bound == None:
            vol_bound = self.vol_bound

        index = {}
        for fea in self.vol_bound['fea_type']:
            ind = coord2index(inputs.clone(), vol_bound['input_vol'], reso=self.vol_bound['reso'], plane=fea)
            index[fea] = ind.unsqueeze(0)
            input_cur = add_key(inputs.unsqueeze(0), index, 'points', 'index', device=device)

        with torch.no_grad():
            fea, unet = self.model.encode_inputs(input_cur)

        return fea, unet

    def predict_crop_occ(self, pi, c, vol_bound=None, **kwargs):
        ''' Predict occupancy values for a crop

        Args:
            pi (dict): query points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        '''
        occ_hat = pi.new_empty((pi.shape[0]))

        if pi.shape[0] == 0:
            return occ_hat
        pi_in = pi.unsqueeze(0)
        pi_in = {'p': pi_in}
        p_n = {}
        for key in self.vol_bound['fea_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(pi.clone(), vol_bound['input_vol'], plane=key).unsqueeze(0).to(self.device)
        pi_in['p_n'] = p_n

        # predict occupancy of the current crop
        with torch.no_grad():
            occ_cur = self.model.decode(pi_in, c, **kwargs).logits
        occ_hat = occ_cur.squeeze(0)

        return occ_hat

    def eval_points(self, p, c=None, vol_bound=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            occ_hat = self.predict_crop_occ(pi, c, vol_bound=vol_bound, **kwargs)
            occ_hats.append(occ_hat)

        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

    def generate_mesh_from_neural_map(self, latent_all, return_stats=True):
        self.model.eval()
        device = self.device
        decoder_unet = self.unet.decoders

        n_crop = self.vol_bound['n_crop']
        print("Decoding latent codes from {} voxels".format(n_crop))
        # acquire the boundary for every crops
        self.get_crop_bound(self.pointcloud_all)
        kwargs = {}
        stats_dict = {}

        n = self.resolution0

        n_crop_axis = self.vol_bound['axis_n_crop']
        max_x, max_y, max_z = n_crop_axis[0], n_crop_axis[1], n_crop_axis[2]
        value_grid = np.zeros((max_x, max_y, max_z, n, n, n))

        if self.bound_interpolation:
            overlap = self.vol_bound['input_crop_size'] / self.vol_bound['query_crop_size']
            overlap = overlap - 1.0
            overlap = overlap / 2 # 1 side
            n_overlap = round(n*overlap*0.5)
            ov = n_overlap
            n_query_overlap = n_overlap * 2 + n
            n_interpolate = n_query_overlap - n
            weights = np.linspace(0.0, 1.0, num=n_interpolate)
            value_grid_overlap = np.zeros((max_x, max_y, max_z, n_query_overlap, n_query_overlap, n_query_overlap))
            visited = np.zeros((max_x, max_y, max_z), dtype = int)

        for i in trange(n_crop):
            # 1D to 3D index
            x = i // (max_y * max_z)
            remainder_x = i % (max_y * max_z)
            y = remainder_x // max_z
            z = remainder_x % max_z

            latent = latent_all[x, y, z].unsqueeze(0)

            if torch.sum(latent) == 0.0:
                continue
            else:
                c = {}
                c['grid'] = self.unet3d_decode(latent, decoder_unet)
                c['grid'] = self.unet.final_conv(c['grid'])

                # encode the current crop
                vol_bound = {}
                vol_bound['query_vol'] = self.vol_bound['query_vol'][i]
                vol_bound['input_vol'] = self.vol_bound['input_vol'][i]

                bb_min = self.vol_bound['query_vol'][i][0]
                bb_max = bb_min + self.vol_bound['query_crop_size']

                t = (bb_max - bb_min) / n  # interval

                if self.bound_interpolation:
                    bb_min = bb_min - t*n_overlap
                    bb_max = bb_max + t*n_overlap - 0.00000000001 # correct floating precision

                    pp = np.mgrid[bb_min[0]:bb_max[0]:t[0], bb_min[1]:bb_max[1]:t[1], bb_min[2]:bb_max[2]:t[2]]
                    pp = pp.reshape(3,-1).T
                    pp = torch.from_numpy(pp).to(device)

                    values_overlap = self.eval_points(pp, c, vol_bound=vol_bound, **kwargs).detach().cpu().numpy()
                    values_overlap = values_overlap.reshape(n_query_overlap, n_query_overlap, n_query_overlap)

                    # convert to probability here
                    values_overlap = np.exp(values_overlap) / (1 + np.exp(values_overlap))

                    value_grid_overlap[x, y, z] = values_overlap
                    value_grid[x, y, z] = values_overlap[n_overlap:-n_overlap, n_overlap:-n_overlap, n_overlap:-n_overlap]
                    visited[x, y, z] = 1

                    # Interpolate x-axis - right
                    if (x + 1) < max_x:
                        if visited[x + 1, y, z] == 1:
                            overlap_A = value_grid[x, y, z, -ov:, :, :]
                            overlap_B = value_grid_overlap[x+1, y, z, :ov, n_overlap:-n_overlap, n_overlap:-n_overlap]
                            overlap_A = (overlap_A.T * weights[-ov:][::-1]).T
                            overlap_B = (overlap_B.T * weights[:ov]).T

                            value_grid[x, y, z, -ov:, :, :] = overlap_A + overlap_B

                            overlap_A = value_grid_overlap[x, y, z, -ov:, n_overlap:-n_overlap, n_overlap:-n_overlap]
                            overlap_B = value_grid[x+1, y, z, :ov, :, :]
                            overlap_A = (overlap_A.T * weights[:ov][::-1]).T
                            overlap_B = (overlap_B.T * weights[ov:]).T

                            value_grid[x+1 , y, z, :ov, :, :] = overlap_A + overlap_B

                    # Interpolate x-axis - left
                    if (x - 1) >= 0:
                        if visited[x - 1, y, z] == 1:
                            overlap_A = value_grid[x, y, z, :ov, :, :]
                            overlap_B = value_grid_overlap[x-1, y, z, -ov:, n_overlap:-n_overlap, n_overlap:-n_overlap]
                            overlap_A = (overlap_A.T * weights[-ov:]).T
                            overlap_B = (overlap_B.T * weights[:ov][::-1]).T

                            value_grid[x, y, z, :ov, :, :] = overlap_A + overlap_B

                            overlap_A = value_grid_overlap[x, y, z, :ov, n_overlap:-n_overlap, n_overlap:-n_overlap]
                            overlap_B = value_grid[x-1, y, z, -ov:, :, :]
                            overlap_A = (overlap_A.T * weights[:ov]).T
                            overlap_B = (overlap_B.T * weights[-ov:][::-1]).T

                            value_grid[x-1, y, z, -ov:, :, :] = overlap_A + overlap_B

                    # Interpolate y-axis - up
                    if (y + 1) < max_y:
                        if visited[x, y+1, z] == 1:
                            overlap_A = value_grid[x, y, z, :, -ov:, :]
                            overlap_B = value_grid_overlap[x, y+1, z, n_overlap:-n_overlap, :ov, n_overlap:-n_overlap]
                            overlap_A = (overlap_A.transpose(2,0,1) * weights[-ov:][::-1]).transpose(1,2,0)
                            overlap_B = (overlap_B.transpose(2,0,1) * weights[:ov]).transpose(1,2,0)

                            value_grid[x, y, z, :, -ov:, :] = overlap_A + overlap_B

                            overlap_A = value_grid_overlap[x, y, z, n_overlap:-n_overlap, -ov:, n_overlap:-n_overlap]
                            overlap_B = value_grid[x, y+1, z, :, :ov, :]
                            overlap_A = (overlap_A.transpose(2,0,1) * weights[:ov][::-1]).transpose(1,2,0)
                            overlap_B = (overlap_B.transpose(2,0,1) * weights[-ov:]).transpose(1,2,0)

                            value_grid[x, y+1, z, :, :ov, :] = overlap_A + overlap_B

                    # Interpolate y-axis - down
                    if (y - 1) >= 0:
                        if visited[x, y-1, z] == 1:
                            overlap_A = value_grid[x, y, z, :, :ov, :]
                            overlap_B = value_grid_overlap[x, y-1, z, n_overlap:-n_overlap, -ov:, n_overlap:-n_overlap]
                            overlap_A = (overlap_A.transpose(2,0,1) * weights[-ov:]).transpose(1,2,0)
                            overlap_B = (overlap_B.transpose(2,0,1) * weights[:ov][::-1]).transpose(1,2,0)

                            value_grid[x, y, z, :, :ov, :] = overlap_A + overlap_B

                            overlap_A = value_grid_overlap[x, y, z, n_overlap:-n_overlap, :ov, n_overlap:-n_overlap]
                            overlap_B = value_grid[x, y-1, z, :, -ov:, :]
                            overlap_A = (overlap_A.transpose(2,0,1) * weights[:ov]).transpose(1,2,0)
                            overlap_B = (overlap_B.transpose(2,0,1) * weights[-ov:][::-1]).transpose(1,2,0)

                            value_grid[x, y-1, z, :, -ov:, :] = overlap_A + overlap_B

                    # Interpolate z-axis - front
                    if (z + 1) < max_z:
                        if visited[x, y, z+1] == 1:
                            overlap_A = value_grid[x, y, z, :, :, -ov:]
                            overlap_B = value_grid_overlap[x, y, z+1, n_overlap:-n_overlap, n_overlap:-n_overlap, :ov]
                            overlap_A = overlap_A * weights[-ov:][::-1]
                            overlap_B = overlap_B.transpose * weights[:ov]

                            value_grid[x, y, z, :, :, -ov:] = overlap_A + overlap_B

                            overlap_A = value_grid_overlap[x, y, z, n_overlap:-n_overlap, n_overlap:-n_overlap, -ov:]
                            overlap_B = value_grid[x, y, z+1, :, :, :ov]
                            overlap_A = overlap_A * weights[:ov][::-1]
                            overlap_B = overlap_B * weights[-ov:]

                            value_grid[x, y, z+1, :, :, :ov] = overlap_A + overlap_B

                    # Interpolate z-axis - back
                    if (z - 1) >= 0:
                        if visited[x, y, z - 1] == 1:
                            overlap_A = value_grid[x, y, z, :, :, :ov]
                            overlap_B = value_grid_overlap[x, y, z-1, n_overlap:-n_overlap, n_overlap:-n_overlap, -ov:]
                            overlap_A = overlap_A * weights[-ov:]
                            overlap_B = overlap_B * weights[:ov][::-1]

                            value_grid[x, y, z, :, :, :ov] = overlap_A + overlap_B

                            overlap_A = value_grid_overlap[x, y, z, n_overlap:-n_overlap, n_overlap:-n_overlap, :ov]
                            overlap_B = value_grid[x, y, z - 1, :, :, -ov:]
                            overlap_A = overlap_A * weights[:ov]
                            overlap_B = overlap_B * weights[-ov:][::-1]

                            value_grid[x, y, z-1, :, :, -ov:] = overlap_A + overlap_B
                else:
                    pp = np.mgrid[bb_min[0]:bb_max[0]:t[0], bb_min[1]:bb_max[1]:t[1], bb_min[2]:bb_max[2]:t[2]]
                    pp = pp.reshape(3, -1).T
                    pp = torch.from_numpy(pp).to(device)

                    values = self.eval_points(pp, c, vol_bound=vol_bound, **kwargs).detach().cpu().numpy()
                    values = values.reshape(n, n, n)

                    # convert to probability here
                    values = np.exp(values) / (1 + np.exp(values))
                    value_grid[x, y, z] = values

        print("Organize voxels for mesh generation")

        r = n * 2 ** self.upsampling_steps
        occ_values = np.array([]).reshape(r, r, 0)
        occ_values_y = np.array([]).reshape(r, 0, r * n_crop_axis[2])
        occ_values_x = np.array([]).reshape(0, r * n_crop_axis[1], r * n_crop_axis[2])

        for i in trange(n_crop):
            index_x = i // (max_y * max_z)
            remainder_x = i % (max_y * max_z)
            index_y = remainder_x // max_z
            index_z = remainder_x % max_z

            values = value_grid[index_x][index_y][index_z]

            occ_values = np.concatenate((occ_values, values), axis=2)
            # along y axis
            if (i + 1) % n_crop_axis[2] == 0:
                occ_values_y = np.concatenate((occ_values_y, occ_values), axis=1)
                occ_values = np.array([]).reshape(r, r, 0)
            # along x axis
            if (i + 1) % (n_crop_axis[2] * n_crop_axis[1]) == 0:
                occ_values_x = np.concatenate((occ_values_x, occ_values_y), axis=0)
                occ_values_y = np.array([]).reshape(r, 0, r * n_crop_axis[2])

        value_grid = occ_values_x
        value_grid[np.where(value_grid == 1.0)] = 0.9999999
        value_grid[np.where(value_grid == 0.0)] = 0.0000001
        value_grid = np.log(value_grid / (1 - value_grid))

        print("Generating mesh")
        t0 = time.time()
        mesh = self.extract_mesh(value_grid, stats_dict=stats_dict)
        t1 = time.time()
        generate_mesh_time = t1 - t0
        print("Mesh generated in {:.2f}s".format(generate_mesh_time))
        if return_stats:
            return mesh, stats_dict, value_grid
        else:
            return mesh


    def unet3d_decode(self, z, decoder_unet):
        ''' Decode latent code into feature volume

        Args:
            z (torch.Tensor): latent code
            decoder_unet (torch model): decoder
        '''

        batch_size, num_channels, h, w, d = z.size()

        with torch.no_grad():
            for depth in range(len(decoder_unet)):
                up = (depth + 1) * 2
                dummy_shape_tensor = torch.zeros(1, 1, h * up, w * up, d * up)
                z = decoder_unet[depth].upsampling(dummy_shape_tensor, z)
                z = decoder_unet[depth].basic_module(z)

        return z

    def extract_mesh(self, occ_hat, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1

        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (
                        self.vol_bound['axis_n_crop'].max() * self.resolution0 * 2 ** self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else:
            # Normalize to bounding box
            vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
            vertices = box_size * (vertices - 0.5)

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=None,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        return mesh
