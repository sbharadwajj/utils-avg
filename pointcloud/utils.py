import numpy as np

# from open3d import read_point_cloud, draw_geometries, PointCloud, Vector3dVector, io
# from open3d.geometry import get_center, rotate
import struct
import open3d as o3d

def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd

def compute_center(cloud):
    center = (cloud.get_center()).astype(np.float64)
    return center[...,np.newaxis]

def crop_point(cloud, r, ref):
    points = np.asarray(cloud.points).astype(np.float64)
    dist = np.linalg.norm(np.transpose(ref) - points, axis=1)
    bounding_box = points[(dist < r)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bounding_box)
    return pcd

def normalize(cloud, ip_cloud):
    gt = np.asarray(cloud.points).astype(np.float64)
    ip = np.asarray(ip_cloud.points).astype(np.float64)
    import pdb; pdb.set_trace()

def viz_pointcloud(path, poses, ip_pcd, r):
    cloud = o3d.io.read_point_cloud(path) # Read the point cloud
    rot = poses[:,:3].astype(np.float64)
    translation = poses[:, 3:].astype(np.float64)
    normalize(cloud, ip_pcd)
    cropped = crop_point(cloud, r, translation) # crop the point cloud`
    o3d.visualization.draw_geometries([cropped]) # Visualize the point cloud
    o3d.visualization.draw_geometries([ip_transformed])

def only_viz(path, path_inp):
    o3d.visualization.draw_geometries([path])
    o3d.visualization.draw_geometries([path_inp])

if __name__ == "__main__":
    r = 100
    path = 'data_3d_semantics/2013_05_28_drive_0000_sync/static/001270_001549.ply'
    path_velodyne = 'KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000001270.bin'
    euclidean_ball = np.asarray([(0.3149843591, 0.9483212912, 0.0383612069,1240.090749), (0.949085342, -0.3149222883,-0.0078080719, 3937.974735), (0.0046762383 ,0.0388674797, -0.9992334321, 115.4630822)])
    pcd = convert_kitti_bin_to_pcd(path_velodyne)
    viz_pointcloud(path, euclidean_ball, pcd, r)
    # import pdb; pdb.set_trace()
    # o3d.io.write_point_cloud("test1.ply", pcd)
    # viz_pointcloud('/home/shrisha/masters/WS-20/AVG/data/test1.pcd', euclidean_ball)
    