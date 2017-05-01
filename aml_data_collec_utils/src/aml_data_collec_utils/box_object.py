import tf
import rospy
import random
import numpy as np
from tf import TransformListener
from sensor_msgs.msg import Image
from aml_data_collec_utils.config import config
from aml_visual_tools.visual_tools import show_image
from aml_io.convert_tools import rosimage2openCVimage
from aml_perception.camera_sensor import CameraSensor
from ros_transform_utils import get_pose, transform_to_pq, pq_to_transform

class BoxObject(object):

    def __init__(self):

        self._tf              = TransformListener()

        self._dimensions      = np.array([config['box_type']['length'], config['box_type']['height'], config['box_type']['breadth']])

        self._frame_name      = 'box'

        self._base_frame_name = 'base'

        self._camera_sensor =  CameraSensor()
        
        self._box_reset_pos0 = None

        self._last_pushes = None

        # Publish
        self._br = tf.TransformBroadcaster()

        update_rate = 30.0
        update_period = rospy.Duration(1.0/update_rate)
        rospy.Timer(update_period, self.update_frames)

    #this is a util that makes the data in storing form
    def get_effect(self):

        status = {}

        try:
            pose, _, _   = self.get_pose()
            p, q   = transform_to_pq(pose)
            
            status['box_pos'] = p
            #all the files in package follows np.quaternion convention, that is
            # w,x,y,z while ros follows x,y,z,w convention
            status['box_ori'] = np.array([q[3],q[0],q[1],q[2]])

            status['box_tracking_good'] = True
        except:
            print "tracking failed"
            status['box_pos'] = np.zeros(3)
            status['box_ori'] = np.zeros(3)
            status['box_tracking_good'] = False

        return status


    def get_pre_push(self,idx):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        return get_pose(self._tf,self._base_frame_name,'pre_push%d'%(idx,), time)

    def get_push_pose(self):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        return get_pose(self._tf,self._base_frame_name,'pre_push%d'%(idx,), time)

    def get_reset_pose(self):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        # Box pose
        pose, _, _ = self.get_pose()

        p, q = transform_to_pq(pose)

        reset_offset = config['reset_spot_offset']
        tip_offset = config['end_effector_tip_offset']
        pos_rel_box = np.array([reset_offset[0],reset_offset[1]+tip_offset[1],reset_offset[2],1])
        pos = np.asarray(np.dot(pose,pos_rel_box)).ravel()[:3]

        return pos, q

    def update_frames(self,event):
        try:
            if self._last_pushes is not None:
                pushes = self._last_pushes
                count = 0
                for push in pushes:

                    now = rospy.Time.now()
                    for pose in push['poses']:

                        self._br.sendTransform(pose['pos'], pose['ori'], now, "%s%d"%(push['name'],count), 'base')
                        count += 1


        except Exception as e:
            print "Error on update frames", e
            pass

    def get_pose(self, time = None, time_out=5.):
        start_time = rospy.Time.now()
        timeout = rospy.Duration(time_out) # Timeout of 'time_out' seconds
        if time is None:
            while (time is None):
                try:
                    time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)
                except Exception as e:
                    if (rospy.Time.now() - start_time > timeout):
                        # time = rospy.Time.now()
                        raise Exception("Time out reached, was not able to get *tf.getLatestCommonTime*")
                        break
                    else:
                        continue

        box_pos = box_q = None
        # Getting box center (adding center offset to retrieved pose)
        pose = get_pose(self._tf, self._base_frame_name, self._frame_name, time)
        box_center_offset = config['box_center_offset']
        box_pos = np.asarray(np.dot(pose,np.array([box_center_offset[0],box_center_offset[1],box_center_offset[2],1]))).ravel()[:3]
        _, box_q = transform_to_pq(pose)
        pose = pq_to_transform(self._tf,box_pos,box_q)

        return pose, box_pos, box_q

    def get_curr_image(self, time=None, time_out=5.):
        start_time = rospy.Time.now()
        timeout = rospy.Duration(time_out) # Timeout of 'time_out' seconds
        
        _scene_image = None
        
        while _scene_image is None:
            _scene_image = self._camera_sensor._curr_rgb_image
            if (rospy.Time.now() - start_time > timeout):
                raise Exception("Time out reached, was not able to get *_scene_image*")
                break
            else:
                continue
        # _scene_image = np.transpose(_scene_image, axes=[2,1,0]).flatten()       

        '''
        NOTE:  _scene_image is None in the beginning, we need to make sure that 
        when it is read for the second time, it is the currect scene and not an old image,
        that is we need a way to invalidate the existing image, after it is read
        '''
        return _scene_image

    # Computes a list of "pushes", a push contains a pre-push pose, 
    # a push action (goal position a push starting from a pre-push pose) 
    # and its respective name
    # It also returns the current box pose, and special reset_push
    def get_pushes(self, use_random=True):

        success = False
        max_trials = 200
        trial_count = 0

        pos = pose = time = ee_pose = box_q = box_pos =  None

        while trial_count < max_trials and not success:

            try:
                pose, box_pos, box_q = self.get_pose()

                time = self._tf.getLatestCommonTime(self._base_frame_name, 'left_gripper')
                ee_pose = get_pose(self._tf, self._base_frame_name,'left_gripper', time)

                ee_pos, q_ee = transform_to_pq(ee_pose)

                reset_pos, reset_q = self.get_reset_pose()

                success = True
            except Exception as e:
                print "Failed to get required transforms", e
                trial_count += 1


        
        if success:
            pre_push_offset = config['pre_push_offsets']

            length_div2 = config['box_type']['length']*config['scale_adjust']/2.0
            breadth_div2 = config['box_type']['breadth']*config['scale_adjust']/2.0
            
            if use_random:
                x_box = random.uniform(-length_div2,length_div2) # w.r.t box frame
                z_box = random.uniform(-breadth_div2,breadth_div2) # w.r.t box frame
            else:
                x_box = 0.
                z_box = 0.

            pre_positions = np.array([[pre_push_offset[0]    , pre_push_offset[1],  z_box,               1], # right-side of the object
                                  [-pre_push_offset[0]   , pre_push_offset[1],  z_box,               1], # left-side of the object
                                  [x_box                 , pre_push_offset[1],  pre_push_offset[2],  1], # front of the object
                                  [x_box                 , pre_push_offset[1], -pre_push_offset[2],  1]])  # back of the object

            push_locations = np.array([[0    , pre_push_offset[1],  z_box,               1], # right-side of the object
                                       [0   , pre_push_offset[1],  z_box,               1], # lef-side of the object
                                       [x_box                 , pre_push_offset[1],  0,  1], # front of the object
                                       [x_box                 , pre_push_offset[1],  0,  1]])  # back of the object


            pushes = []
            count = 0
            for pos_idx in range(len(pre_positions)):

                pre_position = pre_positions[pos_idx] # w.r.t to box
                push_position = push_locations[pos_idx] # w.r.t to box

                # "position" is relative to the box        
                pre_push_pos1 = np.asarray(np.dot(pose,pre_position)).ravel()[:3]
                pre_push_dir0 = pre_push_pos1 - ee_pos
                pre_push_dir0[2] = 0
                pre_push_pos0 = ee_pos + pre_push_dir0

                # Pushing towards the center of the box
                push_action = np.asarray(np.dot(pose,push_position)).ravel()[:3] # w.r.t to base frame now

                push_xz = np.array([push_position[0],push_position[2]])
                pushes.append({'poses': [{'pos': pre_push_pos0, 'ori': box_q}, {'pos': pre_push_pos1, 'ori': box_q}], 'push_action': push_action, 'push_xz': push_xz, 'name' : 'pre_push%d'%(count,)})

                count += 1


            if self._box_reset_pos0 is None:
                self._box_reset_pos0 = reset_pos

            # Reset push is a special kind of push
            
            reset_offset = config['reset_spot_offset']
            pre_reset_offset = config['pre_reset_offsets']
            pos_rel_box = np.array([reset_offset[0],reset_offset[1]+pre_reset_offset[1],reset_offset[2],1])
            pre_reset_pos = np.asarray(np.dot(pose,pos_rel_box)).ravel()[:3]

            reset_displacement = (self._box_reset_pos0 - reset_pos)
            reset_push = {'poses': [{'pos': pre_reset_pos, 'ori': reset_q}, {'pos': reset_pos, 'ori': reset_q}], 'push_action': reset_displacement, 'name' : 'reset_spot'}
            

            # pushes.append(reset_push)

            return pushes, pose, reset_push
        else:
            return [], None, None