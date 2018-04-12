import pybullet as pb

def setup_bullet_visualizer(render=True):

    cid = pb.connect(pb.SHARED_MEMORY)
        
    if (cid < 0) and render:
        
        cid = pb.connect(pb.GUI)
    
        pb.resetDebugVisualizerCamera(-6.3, -180,-41, [0.52,-0.2,-0.33])

    else:
        
        cid = pb.connect(pb.DIRECT)

    return cid
