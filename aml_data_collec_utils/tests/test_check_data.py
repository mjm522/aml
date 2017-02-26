import sys
import cv2
import aml_robot
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from aml_data_collec_utils.record_sample import Sample
from cv_bridge import CvBridge, CvBridgeError



# def show_depth_image(depth_image):

#     try:
#          # print("Max",max_val,"Min",min_val)
#     #     # cv_image.convertTo(B,CV_8U,255.0/(Max-Min));
#         img = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
#         cv2.imshow("Depth Image window", img)
#     except CvBridgeError as e:
#         print(e)


def show_image(image):

    cv2.imshow("RGB Image window", image)

    cv2.waitKey(0)

def plot_demo_data(traj):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    plt.title('Task space trajectories')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(traj[0][0], traj[0][1], traj[0][2],  linewidths=20, color='r', marker='*')
    ax.scatter(traj[-1][0],traj[-1][1],traj[-1][2], linewidths=20, color='g', marker='*')
    ax.plot(traj[:,0],traj[:,1],traj[:,2])

    plt.show()


def visualize_data(data):

    print "Visualizind data with id: \t",  data['sample_id']

    num_data_points = len(data['state'])

    traj = []

    for k in range(num_data_points):
        
        image_rgb = data['state'][k]['rgb_image']
        
        image_depth = data['state'][k]['depth_image']

        traj.append(data['task_effect'][k]['pos'])

        show_image(image_rgb)
        # show_depth_image(image_depth)

    plot_demo_data(np.asarray(traj).squeeze())
    
    
def main(sample_idx):

    sample = Sample()

    if not sample_idx:

        sample_idx = 0 
    
    read_sample_success = True

    while read_sample_success:

        try:
            data = sample.get_sample(sample_id=sample_idx)
        except Exception as e:
            print "Unable to read sample number \t", sample_idx
            read_sample_success = False

        if read_sample_success:
            
            visualize_data(data)

            print "Enter to continue ..."

            sample_idx += 1

    print "Exiting ..."

if __name__ == '__main__':
    main(int(sys.argv[1]))
