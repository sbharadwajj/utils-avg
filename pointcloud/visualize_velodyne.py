import open3d as o3d 
import numpy as np 

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def downsample(inputpcd, voxel_size):
    downpcd = inputpcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downpcd.points).astype(np.float64)

points = np.loadtxt("baseline_data/DATA/downsampled_fused/downsampled_train/400.dat")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)  
down_pcd = resample_pcd(downsample(pcd, 0.4), 4098)
down = o3d.geometry.PointCloud()
down.points = o3d.utility.Vector3dVector(down_pcd)
o3d.visualization.draw_geometries([pcd])
print(pcd)
o3d.visualization.draw_geometries([down])
print(down)
