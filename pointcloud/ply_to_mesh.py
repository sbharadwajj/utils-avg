import numpy as np
import pyvista as pv
import open3d as o3d
import os
import sys
from pyntcloud import PyntCloud

if __name__ == "__main__":
    input_folder = sys.argv[1]
    assert(os.path.exists(input_folder))

    for f in os.listdir(input_folder):
        input_path = os.path.join(input_folder, f)
        pcd = o3d.io.read_point_cloud(input_path)
        # cloud = pv.PolyData(np.asarray(pcd.points))
        # # cloud.plot()

        # volume = cloud.delaunay_3d(alpha=.2)
        # shell = volume.extract_geometry()
        # shell.save(f.split(".")[0] + ".ply")
        # shell.plot()



        # TO PyVista
        cloud = PyntCloud.from_file(input_path)
        converted_triangle_mesh = cloud.to_instance("pyvista", mesh=True)
        converted_triangle_mesh.save(f.split(".")[0] + ".ply")
        # converted_triangle_mesh.to_file(f.split(".")[0] + ".obj", internal=["mesh"])
