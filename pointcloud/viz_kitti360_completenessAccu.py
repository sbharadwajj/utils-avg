'''
else:
    if norm_accu == None:
        colors_accu, vmin, vmax, norm_accu = get_color(accu_vec, None)
        colors_comp, vmin_c, vmax_c, norm_comp = get_color(comp_vec, None)
    else:
        colors_accu, vmin, vmax, norm_accu = get_color(accu_vec, norm_accu)
        colors_comp, vmin_c, vmax_c, norm_comp = get_color(comp_vec, norm_accu)    
print(accu_vec.sum()/8192)
print(comp_vec.sum()/8192)
fig, ax = plt.subplots()
im = ax.imshow(np.diag(comp_vec), cmap='rainbow', vmin=0.0, vmax=6.37)
fig.colorbar(im, ax=ax)
plt.show()
'''
import open3d as o3d 
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from sklearn.preprocessing import normalize

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False

def points_pcd_color(point_set, colors_vec):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    pcd.colors = o3d.utility.Vector3dVector(colors_vec)
    return pcd

def points_pcd(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set) 
    return pcd

def get_thresh(points_z, percent, array_pcd, axis):
    thresh = np.quantile(points_z, percent)
    print(thresh)
    bottom = array_pcd[array_pcd[:, axis] < thresh]
    top = array_pcd[array_pcd[:, axis] > thresh]
    print(bottom.shape)
    print(top.shape)
    return bottom, top

def get_color(accu_vec, norm):
    lower = accu_vec.min()
    upper = accu_vec.max()
    
    if norm == None:
        norm = plt.Normalize(vmin=lower, vmax=upper)
        colors = plt.cm.rainbow(norm(accu_vec))
    else:
        colors = plt.cm.rainbow(norm(accu_vec))
    return colors[:,:3], lower, upper, norm

def get_color_threshold(accu_vec, thresh):
    accu_norm = (accu_vec-accu_vec.min())/(accu_vec.max()-accu_vec.min())
    colors = np.zeros((8192, 3))
    colors[accu_norm < thresh] = (0, 1, 0) # lesser than threshold - green
    colors[accu_norm > thresh] = (1, 0, 0)

    # quantitative
    quant = accu_norm[accu_norm < thresh]
    quant_norma = quant.sum() / quant.shape[0]
    return colors, quant_norma

get_thresh = True
feat = False
path = sys.argv[1]
# path2 = sys.argv[2]
save_path = path + 'kitti360' 
if not os.path.exists(save_path):
    os.mkdir(save_path)
# import pdb; pdb.set_trace()
def plot_scene(path, norm_accu):
    batch = np.load(path)

    preds = batch['predictions']
    inp = batch['data']
    gt = batch['gt']
    completeness = batch['chamx']
    accuracy = batch['chamy']

    accu_files = []
    comp_files = []

    cnt = 0
    for pred, ip, gt, comp, accu in zip(preds, inp, gt, completeness, accuracy): 
        accu_list = []
        comp_list = []
        thresh_list = []
        for thresh in range(1):  
            # print(thresh/10)
            accu_vec = ((accu)*100).astype(float)
            comp_vec = ((comp)*100).astype(float)
            # colors_accu, quant_accu = get_color_threshold(accu_vec, thresh/10)
            # colors_comp, quant_comp  = get_color_threshold(comp_vec, thresh/10)            
            colors_accu, vmin, vmax, norm_accu = get_color(accu_vec, None)
            colors_comp, vmin_c, vmax_c, norm_comp = get_color(comp_vec, None)
            # accu_list.append(quant_accu)
            # comp_list.append(quant_comp)
            thresh_list.append(thresh/10)

            pcd_accu = points_pcd_color(pred, colors_accu)
            pcd_inp = points_pcd(ip.transpose(1,0))
            pcd_comp = points_pcd_color(gt, colors_comp)


            o3d.visualization.draw_geometries([pcd_comp])
            o3d.visualization.draw_geometries([pcd_accu])

            # print(quant_accu*1000)
            # print(quant_comp*1000)

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd_accu)
            vis.update_geometry(pcd_accu)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(os.path.join(save_path, str(cnt) + "_thresh"+ str(thresh) + "_accu.png"))
            vis.destroy_window()

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd_comp)
            vis.update_geometry(pcd_comp)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(os.path.join(save_path, str(cnt) + "_thresh" + str(thresh) + "_comp.png"))
            vis.destroy_window()
            
        accu_files.append(accu_list)
        comp_files.append(comp_list)

        cnt+=1
        # if cnt > 3:
        #     break
    return accu_files, comp_files

accu, comp = plot_scene(path, None)

# fig, ax = plt.subplots(1, 2, figsize=(20, 5))

# ax[0].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], accu[0], label="0")
# ax[0].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], accu[1], label="1")
# ax[0].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], accu[2], label="2")
# ax[0].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], accu[3], label="3")
# # ax[0].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], accu[4], label="4")

# # ax[0].set_title('Accuracy plot')
# ax[0].set_xlabel("Threshold")
# ax[0].set_ylabel("Accuracy")
# ax[0].legend()
# # plt.savefig(save_path + "/accu.png")

# ax[1].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], comp[0], label="0")
# ax[1].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], comp[1], label="1")
# ax[1].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], comp[2], label="2")
# ax[1].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], comp[3], label="3")
# # ax[1].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], comp[3], label="4")

# # ax[1].set_title('Completeness plot')
# ax[1].legend()
# ax[1].set_xlabel("Threshold")
# ax[1].set_ylabel("Completeness")
# fig.savefig(save_path + "/accu_comp.png")

def plot_scene_colors(path, norm_accu):
    batch = np.load(path)

    preds = batch['predictions']
    inp = batch['data']
    gt = batch['gt']
    completeness = batch['chamx']
    accuracy = batch['chamy']

    accu_files = []
    comp_files = []

    cnt = 0
    for pred, ip, gt, comp, accu in zip(preds, inp, gt, completeness, accuracy): 
        accu_list = []
        comp_list = []
        accu_vec = ((accu)*100).astype(float)
        comp_vec = ((comp)*100).astype(float)
        if norm_accu == None:
            colors_accu, vmin, vmax, norm_accu = get_color(accu_vec, None)
            colors_comp, vmin_c, vmax_c, norm_comp = get_color(comp_vec, None)
        else:
            colors_accu, vmin, vmax, norm_accu = get_color(accu_vec, norm_accu)
            colors_comp, vmin_c, vmax_c, norm_comp = get_color(comp_vec, norm_accu)  
        pcd_accu = points_pcd_color(pred, colors_accu)
        pcd_comp = points_pcd_color(gt, colors_comp)
            
    print(accu_vec.sum()/8192)
    print(comp_vec.sum()/8192)
    fig, ax = plt.subplots()
    im = ax.imshow(np.diag(comp_vec), cmap='rainbow', vmin=0.0, vmax=6.37)
    fig.colorbar(im, ax=ax)
    plt.show()