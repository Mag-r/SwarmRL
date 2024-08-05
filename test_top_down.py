import swarmrl as srl

from swarmrl.observables.top_down_image import TopDownImage
import numpy as np
import matplotlib.pyplot as plt
from swarmrl.components import Colloid
import open3d as o3d

colloids = []
for i in range(10):
    colloids.append(Colloid([10000*np.random.rand(),10000*np.random.rand(),0], [np.random.rand(),np.random.rand(),0], i))

    
rafts = o3d.io.read_triangle_mesh("rafts.stl")
obs = TopDownImage(np.array([10000.0, 10000.0]), particle_type=0,custom_mesh=rafts,is_2D=True)
m=obs.compute_observable(colloids=colloids)
m=obs.compute_observable(colloids=colloids)
print(np.shape(m))
plt.imshow(m)
plt.show()