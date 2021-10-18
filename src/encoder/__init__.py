from src.encoder import (
    pointnet
)

encoder_dict = {
    'pointnet_crop_local_pool_latent': pointnet.PatchLocalPoolPointnetLatent,
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
}
