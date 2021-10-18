import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from src.common import normalize_coordinate, normalize_3d_coordinate, map2local, positional_encoding

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.
        This class is used to train the backbone decoder.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, pos_encoding=True):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_encoding==True:
            dim = 60

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # actually trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(
            -1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])

            c = c.transpose(1, 2)

        p = p.float()

        ##################
        if self.pos_encoding:
            pp = self.pe(p)
            net = self.fc_p(pp)
        else:
            net = self.fc_p(p)
        ##################

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

class DecoderLatentToLogits(nn.Module):
    ''' Decoder.
        Decoding latent codes into logits.
        This class is used to train the fusion network.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 padding=0.1, pos_encoding="sin_cos", **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_encoding=="sin_cos":
            dim = 60

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pe = positional_encoding()

    def sample_feature(self, xy, c):
        xy = xy[:, :, None, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        # actually trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def unet3d_decode(self, z, decoder_unet):
        batch_size, num_channels, h, w, d = z.size()

        for depth in range(len(decoder_unet)):
            up = (depth + 1) * 2
            dummy_shape_tensor = torch.zeros(1, 1, h * up, w * up, d * up)
            z = decoder_unet[depth].upsampling(dummy_shape_tensor, z)
            z = decoder_unet[depth].basic_module(z)

        return z

    def forward(self, p, fea, **kwargs):

        decoder_unet = fea['unet3d'].decoders
        p_n = p['p_n']
        p = p['p']

        z = fea['latent']
        z = self.unet3d_decode(z, decoder_unet)
        z = fea['unet3d'].final_conv(z)

        # Get c_prediction
        c = self.sample_feature(p_n['grid'], z)
        c = c.transpose(1,2)

        p = p.float()
        p = self.pe(p)

        # Pass to FC and get logit
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

class PatchLocalDecoder(nn.Module):
    ''' Decoding feature volume to logits.
        This class is used during scene generation.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        local_coord (bool): whether to use local coordinate
        unit_size (float): defined voxel unit size for local system
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='bilinear', local_coord=False, pos_encoding='linear', unit_size=0.1, padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_out = nn.Linear(hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == 'sin_cos':
            self.fc_p = nn.Linear(60, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)
    
    def sample_feature(self, xy, c):
        xy = xy[:, :, None, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        # actually trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        p_n = p['p_n']
        p = p['p']

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_feature(p_n['grid'], c_plane['grid'])

            c = c.transpose(1, 2)

        p = p.float()
        if self.map2local:
            p = self.map2local(p)
        
        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out
