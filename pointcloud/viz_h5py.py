import open3d as o3d 
import numpy as np
import os
import sys
import h5py
import time

global file_name

def h5py_to_pcd(f2):
    data = f2[tuple(f2.keys())[0]]
    arr = data[:]
    return arr

def points_to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  
    return pcd

def rotate_view_gt(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0,0.0)
    ctr.set_zoom(1.5)
    path = "car-chair-gt/" + str(file_name)
    if not os.path.exists(path):
        os.mkdir(path)
    vis.capture_screen_image(path+"/"+str(time.time()) + ".png")
    return False

for f_hdf5 in os.listdir(sys.argv[1]):
    print(f_hdf5)
    file_name = f_hdf5
    path = os.path.join(sys.argv[1], f_hdf5)
    f2 = h5py.File(path, 'r')
    arr = h5py_to_pcd(f2)
    # for pcd_points in arr:
    pcd = points_to_pcd(arr)
    pcd.paint_uniform_color(np.array([0, 0, 1]))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_zoom(1.5)
    vis.register_animation_callback(rotate_view_gt)
    vis.run()
    vis.destroy_window()

    del vis
    del ctr
