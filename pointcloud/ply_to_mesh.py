import open3d as o3d
import trimesh
import numpy as np
import sys
import binrw_py

pcd = o3d.io.read_point_cloud(sys.argv[1])
pcd.estimate_normals()


# estimate radius for rolling ball
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist   

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))

# create the triangular mesh with the vertices and faces from open3d
tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))

trimesh.convex.is_convex(tri_mesh)

mm = trimesh.creation.box()
voxx = mm.voxelized(pitch=0.015)
vx = trimesh.exchange.binvox.export_binvox(voxx)

voxels = binrw_py.read_as_3d_array(vx)
voxels = voxels.data.astype(np.float32)
import pdb;pdb.set_trace()
export("391_mesh.stl")