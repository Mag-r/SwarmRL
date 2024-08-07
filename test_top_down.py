import swarmrl as srl

from swarmrl.observables.top_down_image import TopDownImage
import numpy as np
import matplotlib.pyplot as plt
from swarmrl.components import Colloid
import open3d as o3d
import logging 

logging.basicConfig(level=logging.WARNING)
colloids = []
for i in range(10):
    colloids.append(Colloid([9000*np.random.rand()+500,9000*np.random.rand()+500,0], [np.random.rand(),np.random.rand(),0], i))
# colloids.append(Colloid([0,0,0], [0,0,0], 10))
    
rafts = o3d.io.read_triangle_mesh("rafts.stl")
# rafts=o3d.geometry.TriangleMesh.create_sphere(radius=100) 
obs = TopDownImage(np.array([10000.0, 10000.0, 0.1]), particle_type=0,custom_mesh=rafts,is_2D=True)
m=obs.compute_observable(colloids=colloids)
#
# print(np.shape(m))
# plt.imshow(m, cmap='gray')
plt.imshow(m)
plt.savefig("test_top_down.png")
# plt.show()