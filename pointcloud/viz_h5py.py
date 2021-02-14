import open3d as o3d 
import numpy as np
import os
import sys
import h5py

def h5py_to_pcd(f2):
    data = f2[tuple(f2.keys())[0]]
    arr = data[:]
    return arr

def points_to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  
    return pcd

for f_hdf5 in os.listdir(sys.argv[1]):
    path = os.path.join(sys.argv[1], f_hdf5)
    f2 = h5py.File(path, 'r')
    arr = h5py_to_pcd(f2)
    for pcd_points in arr:
        pcd = points_to_pcd(pcd_points)
        o3d.visualization.draw_geometries([pcd])
