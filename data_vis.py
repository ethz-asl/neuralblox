from src.utils.visualize import visualize_voxels, visualize_pointcloud, visualize_data
from vedo import *
if __name__ == "__main__":
    pcloud = "/home/sam/thesis/asldoc-2022-MT-Samuel-Neural-Planning/data/pcl/livingroom1_1.0_000_000.ply"
    visualize_pointcloud(pcloud)