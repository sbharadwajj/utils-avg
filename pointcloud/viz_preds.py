import open3d as o3d 
import numpy as np
import os
import sys


def points_pcd(point_set, center):
    pcd = o3d.geometry.PointCloud()
    # point_set = point_set - center
    # dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    # point_set = point_set / dist #scale
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd

path = sys.argv[1]
save_path = path + '8-imgs-optim-norm-removed' 
if not os.path.exists(save_path):
    os.mkdir(save_path)
# import pdb; pdb.set_trace()
for path in [path]:
    batch = np.load(path)
    
    for i in range(30):
        center = batch['poses'][i]
        pcd_preds = points_pcd(batch['predictions'][i], center)
        pcd_inp = points_pcd(batch['data'][i], center)
        pcd_gt = points_pcd(batch['gt'][i], center)
        pcd_gt.paint_uniform_color(np.array([1, 0.706, 0]))
        pcd_preds.paint_uniform_color(np.array([0, 1, 0]))
        pcd_inp.paint_uniform_color(np.array([1, 0, 0]))
        o3d.visualization.draw_geometries([pcd_gt, pcd_preds, pcd_inp])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_inp)
        vis.add_geometry(pcd_gt)
        vis.update_geometry(pcd_inp)
        vis.update_geometry(pcd_gt)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(save_path,str(i) + "inp_gt.png"))
        vis.destroy_window()

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_inp)
        vis.add_geometry(pcd_preds)
        vis.update_geometry(pcd_inp)
        vis.update_geometry(pcd_preds)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(save_path,str(i) + "inp_preds.png"))
        vis.destroy_window()

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_gt)
        vis.add_geometry(pcd_preds)
        vis.update_geometry(pcd_gt)
        vis.update_geometry(pcd_preds)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(save_path,str(i) + "gt_preds.png"))
        vis.destroy_window()
        # o3d.visualization.draw_geometries([pcd_preds])
        # o3d.visualization.draw_geometries([pcd_gt])
        # o3d.visualization.draw_geometries([pcd_inp])
        
        