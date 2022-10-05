import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import src.common as common
from vedo import Volume, show, Points


def visualize_data(data, data_type, out_file):
    r''' Visualizes the data with regard to its type.

    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    '''
    if data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)


def visualize_voxels(voxels):
    r''' Visualizes voxel data.

    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    vol = Volume(voxels)
    vol.addScalarBar3D()

    lego = vol.legosurface(vmin=0.01, vmax=1)
    lego.cmap('hot_r', vmin=0.01, vmax=1).addScalarBar3D()

    show(lego).close()


def visualize_pointcloud(pts):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    points = Points(pts)
    show(points, axes=True)
