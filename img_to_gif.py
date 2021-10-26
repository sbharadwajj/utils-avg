import imageio
import sys
import os

path_1 = sys.argv[1]
images = []
sorted_path = sorted(os.listdir(path_1))[:200][0::4]
for filename in sorted_path:
    img = imageio.imread(os.path.join(path_1, filename))
    # import pdb;pdb.set_trace()
    images.append(img)
imageio.mimsave(path_1+'.gif', images)