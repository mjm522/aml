import tf
import copy
import rospy
import numpy as np
import roslib; roslib.load_manifest('visualization_marker_tutorials')
from geometry_msgs.msg import Point 
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


publisher = rospy.Publisher('push_direction', MarkerArray, queue_size=100)


def send_coordinate_frame(br, time, pose):
    #'base' is my parent frame, you can specify with which frame to 
    #see this new frame
    br.sendTransform(pose['pos'], pose['ori'], time, pose['name'], 'base')


def add_new_dim(u):
    u_new = np.zeros([len(u), 3])
    
    for k in range(len(u)):
        u_new[k,0] =  u[k]
        if 0. <= u[k] <= 0.25:
            u_new[k,1] =  0
            u_new[k,2] =  1
        elif 0.25 < u[k] <= 0.5:
            u_new[k,1] =  -1
            u_new[k,2] =  0
        elif 0.5 <= u[k] <= 0.75:
            u_new[k,1] =  0
            u_new[k,2] =  -1
        elif 0.75 < u[k] <= 1.:
            u_new[k,1] =  1
            u_new[k,2] =  0

    return u_new

#other options of primitives available
#check : http://wiki.ros.org/rviz/DisplayTypes/Marker#Line_List_.28LINE_LIST.3D5.29
def send_arrow(arrow_pos):
    
    markerArray = MarkerArray()

    #any number of lines, you can get and append and send them

    marker_link = Marker()

    marker_link.header.frame_id = "base"
    marker_link.type = marker_link.ARROW  
    
    marker_link.scale.x = 0.02#length
    marker_link.scale.y = 0.1#width
    marker_link.scale.z = 0.1 #height

    marker_link.color.a = 1.0
    marker_link.color.r = 0.6
    marker_link.color.g = 0.6
    marker_link.color.b = 0.6

    p_start = Point()
    p_end   = Point()

    p_start.x = arrow_pos['start']['pos'][0]
    p_start.y = arrow_pos['start']['pos'][1]
    p_start.z = arrow_pos['start']['pos'][2]

    p_end.x = arrow_pos['end']['pos'][0]
    p_end.y = arrow_pos['end']['pos'][1]
    p_end.z = arrow_pos['end']['pos'][2]

    marker_link.points.append(p_start)
    marker_link.points.append(p_end)

    markerArray.markers.append(marker_link)

    # Publish the MarkerArray
    publisher.publish(markerArray)


def main():

    rospy.init_node("arrow_node")

    br = tf.TransformBroadcaster()

    #position coordinates
    #quaternion in
    #in x,y,z w format

    start = {'pos': np.array([0.717686359481, 0.354216314659, 0.]),
             'ori': np.array([0.,0.,0.,1.])}

    end = {'pos': np.array([0.717686359481, 0.354216314659, 0.]) + np.array([0., 2.9, 0.]),
           'ori': np.array([0.,0.,0.,1.])}
    
    arrow_pos = {'start': start,
                 'end': end}

    rate = rospy.Rate(100)

    while not rospy.is_shutdown():

        now = rospy.Time.now()

        #now open rviz and find the axis HOHO
        # send_coordinate_frame(br=br, time=now, pose=pose)

        #add marker_array if not selected rviz and select this bit
        send_arrow(arrow_pos)

        rate.sleep()

if __name__ == '__main__':
    main()
