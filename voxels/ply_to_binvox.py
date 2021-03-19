import open3d as o3d 
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import sys
import pandas as pd


from simple_3dviz import Mesh
from simple_3dviz.window import show
from mpl_toolkits.mplot3d import Axes3D 
from pyntcloud import PyntCloud

# import binvox_rw
sys.path.insert(1, '/home/shrisha/masters/WS-20/AVG/data/utils-avg/pointcloud/')
from downsample_pointcloud import downsample

def voxelize(pcd, x, y, z):
    # cloud = PyntCloud.from_file(path)
    cloud = PyntCloud.from_instance("open3d", pcd)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=x, n_y=y, n_z=z)
    voxelgrid = cloud.structures[voxelgrid_id]

    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z
    voxel = np.zeros((x, y, z)).astype(np.bool)

    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxel[x][y][z] = True
    # with open("binvox/00000.binvox", 'wb') as f:
    #     v = binvox_rw.Voxels(voxel, (x, y, z), (0, 0, 0), 1, 'xyz')
    #     v.write(f)
    return voxel, voxelgrid

def voxel_array_to_open3d(voxels_data):
    voxels = []
    vox = o3d.geometry.Voxel()
    vg = o3d.geometry.VoxelGrid()

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    input_folder = sys.argv[1]
    assert(os.path.exists(input_folder))

    for f in os.listdir(input_folder):
        pcd_points = downsample(o3d.io.read_point_cloud(os.path.join(input_folder, f)), 0.4)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)  
        voxels, voxelgrid = voxelize(pcd, 128, 128, 16)
        # voxel_array_to_open3d(voxels)
        if sys.argv[2] == "viz":
            voxelgrid.plot(d=3, mode="density", cmap="hsv")
            # and plot everything
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # ax.voxels(voxels)
            # plt.show()
            # Build a voxel grid from the voxels
            # m = Mesh.from_voxel_grid(
            #     voxels=voxels,
            #     sizes=(0.49,0.49,0.49)
            # )

            # show(
            #     Mesh.from_voxel_grid(voxels=voxels),
            #     light=(-1, -1, 1),
            
            # )
            

