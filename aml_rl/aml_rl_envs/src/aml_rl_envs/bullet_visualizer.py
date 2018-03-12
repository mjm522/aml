import pybullet as pb

def setup_bullet_visualizer():

    cid = pb.connect(pb.SHARED_MEMORY)
        
    if (cid<0):
        
        cid = pb.connect(pb.GUI)
    
        pb.resetDebugVisualizerCamera(-6.3, -180,-41, [0.52,-0.2,-0.33])

    else:
        
        pb.connect(pb.DIRECT)