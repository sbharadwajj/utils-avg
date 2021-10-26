import open3d as o3d 
import numpy as np
import os
import sys
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def rotate_view_gt(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0,0.0)
    ctr.set_zoom(1.5)
    path = "final_baseline_viz/GIF/p++/gt/"
    if not os.path.exists(path):
        os.mkdir(path)
    # vis.capture_screen_image(path+str(time.time())+"gt.png")
    return False

def rotate_view_partial(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0,0.0)
    ctr.set_zoom(1.5)
    path = "final_baseline_viz/GIF/p++/partial/"
    if not os.path.exists(path):
        os.mkdir(path)
    vis.capture_screen_image(path+str(time.time())+"partial.png")
    return False

def rotate_view_pred(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0,0.0)
    ctr.set_zoom(1.5)
    path = "final_baseline_viz/GIF/p++/pred/"
    if not os.path.exists(path):
        os.mkdir(path)
    vis.capture_screen_image(path+str(time.time())+"pred.png")
    return False


def points_pcd(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd

def points_pcd_color(pcd):
    norm = plt.Normalize(vmin=0.0, vmax=0.2)
    pcd_points = np.asarray(pcd.points)
    colors = plt.cm.jet(norm(pcd_points[:, 2]))
    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
    return pcd

def points_pcd_norm(point_set, center):
    pcd = o3d.geometry.PointCloud()
    point_set = point_set - center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #scale
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd

path = sys.argv[1]
save_path = path + '_TESST'
if not os.path.exists(save_path):
    os.mkdir(save_path)
# import pdb; pdb.set_trace()
cnt = 0
# if cnt in [0, 15]:
for npz in os.listdir(path):
    batch = np.load(os.path.join(path,npz))
    
    for i in range(1):#range(len(batch)):
        # center = batch['poses'][i]
        i=0
        pcd_inp_ = (points_pcd(batch['inp'][i])).normalize_normals() # (-1, 1)
        pcd_inp = points_pcd_color(pcd_inp_)
        center = pcd_inp.get_center()

        pred_ = batch['pred'][i]
        pred = torch.sigmoid(torch.from_numpy(pred_)).numpy()
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0   

        gt = batch['gt'][i]


        gt_occ = np.where(gt==1)
        # import pdb;pdb.set_trace()
        pcd_gt_ = points_pcd_norm(np.vstack(gt_occ).transpose(), center)
        pcd_gt = points_pcd_color(pcd_gt_)
        pred_occ = np.where(pred==1)
        pcd_preds_ = points_pcd_norm(np.vstack(pred_occ).transpose(), center)   
        pcd_preds = points_pcd_color(pcd_preds_)
        # import pdb;pdb.set_trace()
        # pcd_gt.paint_uniform_color(np.array([1, 0.706, 0]))
        # pcd_preds.paint_uniform_color(np.array([0, 1, 0]))
        # pcd_inp.paint_uniform_color(np.array([1, 0, 0]))

        # vox_gt = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_gt,voxel_size=1.0)
        # vox_pred = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_preds,1.0)
        '''
        THIS CONVERSION IS CORRECT, THE NUMBER OF OCCUPIED VOXELS IS THE SAME,
        vox size only controls visualization and nothing else
        '''

        # o3d.visualization.draw_geometries([pcd_inp])
        # o3d.visualization.draw_geometries([pcd_preds])
        # o3d.visualization.draw_geometries([pcd_gt])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_gt)
        ctr = vis.get_view_control()
        # ctr.scale(25)
        ctr.set_zoom(1.5)
        vis.register_animation_callback(rotate_view_gt)
        vis.run()
        # vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "gt.png"))
        vis.destroy_window()

        del vis
        del ctr

        vis = o3d.visualization.Visualizer()
        vis.create_window()        
        vis.add_geometry(pcd_preds)
    
        ctr = vis.get_view_control()
        # ctr.scale(25)
        ctr.set_zoom(1.5)
        vis.register_animation_callback(rotate_view_pred)
        vis.run()
        # vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "preds.png"))
        vis.destroy_window()

        del vis
        del ctr

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_inp)

        ctr = vis.get_view_control()
        # ctr.scale(25)
        ctr.set_zoom(1.5)
        vis.register_animation_callback(rotate_view_partial)
        vis.run()
        # vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "partial.png"))
        vis.destroy_window()

        del vis
        del ctr

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd_inp)
        # vis.add_geometry(pcd_gt)
        # ctr = vis.get_view_control()
        # # ctr.scale(25)
        # ctr.set_zoom(1.5)
        # vis.run()
        # vis.capture_screen_image(os.path.join(save_path,npz+str(i) + "INP_GT.png"))
        # vis.destroy_window()

        # del vis
        # del ctr

        cnt+=1
