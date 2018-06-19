import numpy as np
import quaternion as q
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    #this bit was borrowed from: 
    #https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def draw_frame(pos, ori, axes, l=1.):

    #draw a frame at the point pos, which is oriented at ori
    #the function draws a frame where each axis has a length of l
    #axes is a handle to a 3d plot axis e.g.: ax = fig.add_subplot(111, projection='3d'

    if len(ori.flatten()) == 3:
        rot = q.as_rotation_matrix(q.from_euler_angles(ori[0],ori[1],ori[2]))
    elif len(ori.flatten()) == 4:
        rot = q.as_rotation_matrix(q.quaternion(ori[0], ori[1], ori[2], ori[3]))
    elif len(ori.flatten()) == 9:
        rot = ori

    colrs = ['r', 'g', 'b']

    ends = [np.dot(rot, np.array([l,0,0])) + pos, 
            np.dot(rot, np.array([0,l,0])) + pos, 
            np.dot(rot, np.array([0,0,l])) + pos]

    for k in range(3):

        axes.add_artist(Arrow3D([pos[0], ends[k][0]], 
                                 [pos[1], ends[k][1]], 
                                 [pos[2], ends[k][2]],
                                 mutation_scale=20, 
                                 lw=3, arrowstyle="-|>", color=colrs[k]))

##test code
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    draw_frame(pos=np.ones(3)*0.15, ori=(0., np.pi/3, 0), axes=ax, l=0.25)
    plt.show()
