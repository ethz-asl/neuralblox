# from src.encoder import pointnet
from .pointnet import PatchLocalPoolPointnetLatent, LocalPoolPointnet

encoder_dict = {
    'pointnet_crop_local_pool_latent': PatchLocalPoolPointnetLatent,
    'pointnet_local_pool': LocalPoolPointnet,
}
