import os
import rospy
import random
import numpy as np
import quaternion
from config import config_push_world
from aml_robot.mujoco.mujoco_robot import MujocoRobot
from aml_robot.mujoco.mujoco_viewer   import  MujocoViewer
from aml_data_collec_utils.record_sample import RecordSample


class BoxObject():

    def __init__(self, robot_interface=None, config=config_push_world):

        self._dimensions      = np.array([config['box_type']['length'], config['box_type']['height'], config['box_type']['breadth']])

        self._frame_name      = 'box'

        self._base_frame_name = 'base'

        self._box_reset_pos0 = None

        self._last_pushes = None

        self._robot = robot_interface

        update_rate = 30.0
        update_period = rospy.Duration(1.0/update_rate)
        # rospy.Timer(update_period, self.update_frames)

    #this is a util that makes the data in storing form
    def get_effect(self):
        pose, p, q   = self.get_pose()
        status = {}
        status['pos'] = p
        #all the files in package follows np.quaternion convention, that is
        # w,x,y,z while ros follows x,y,z,w convention
        status['ori'] = np.array([q[3],q[0],q[1],q[2]])

        return status


    def get_pre_push(self,idx):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        return get_pose(self._tf,self._base_frame_name,'pre_push%d'%(idx,), time)

    def get_push_pose(self):

        time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        return get_pose(self._tf,self._base_frame_name,'pre_push%d'%(idx,), time)

    def get_reset_pose(self):

        # Box pose
        pose = self._robot._reset_qpos[7:]

        pos = pose[:3]
        q   = pose[3:]

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

    def get_pose(self, time = None):

        # if time is None:
        #     time = self._tf.getLatestCommonTime(self._base_frame_name, self._frame_name)

        # box_pos = box_q = None

        box_pose = self._robot._model.data.qpos[:7].flatten()

        box_pos = box_pose[:3]

        box_q = box_pose[3:]

        rot = quaternion.as_rotation_matrix(np.quaternion(box_q[3], box_q[0], box_q[1], box_q[2]))

        pose = np.vstack([np.hstack([rot, box_pos[:,None]]),np.array([0.,0.,0.,1.])])

        # Getting box center (adding center offset to retrieved pose)
        # pose = get_pose(self._tf, self._base_frame_name,self._frame_name, time)
        # box_center_offset = config_push_world['box_center_offset']
        # box_pos = np.asarray(np.dot(pose,np.array([box_center_offset[0],box_center_offset[1],box_center_offset[2],1]))).ravel()[:3]
        # _, box_q = transform_to_pq(pose)
        # pose = pq_to_transform(self._tf,box_pos,box_q)


        return pose, box_pos.flatten(), box_q

    # Computes a list of "pushes", a push contains a pre-push pose, 
    # a push action (goal position a push starting from a pre-push pose) 
    # and its respective name
    # It also returns the current box pose, and special reset_push
    def get_pushes(self):

        success = False
        max_trials = 200
        trial_count = 0

        pos = pose = time = ee_pose = box_q = box_pos =  None

        ee_pos, q_ee = self._robot.get_ee_pose()

        reset_pos, reset_q = self.get_reset_pose()

        success = True

        pose, box_pos, box_q = self.get_pose()

        # while trial_count < max_trials and not success:

        #     try:
        #         pose, box_pos, box_q = self.get_pose()

        #         time = self._tf.getLatestCommonTime(self._base_frame_name, 'left_gripper')
        #         ee_pose = get_pose(self._tf, self._base_frame_name,'left_gripper', time)

        #         ee_pos, q_ee = transform_to_pq(ee_pose)

        #         reset_pos, reset_q = self.get_reset_pose()

        #         success = True
        #     except Exception as e:
        #         print "Failed to get required transforms", e
        #         trial_count += 1


        
        if success:
            pre_push_offset = config_push_world['pre_push_offsets']

            length_div2 = config_push_world['box_type']['length']/2
            breadth_div2 = config_push_world['box_type']['breadth']/2
            
            x_box = random.uniform(-length_div2,length_div2) # w.r.t box frame
            z_box = random.uniform(-breadth_div2,breadth_div2) # w.r.t box frame

            pre_positions = np.array([[pre_push_offset[0]    , pre_push_offset[1],  z_box,               1], # right-side of the object
                                  [-pre_push_offset[0]   , pre_push_offset[1],  z_box,               1], # lef-side of the object
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
            
            reset_offset = config_push_world['reset_spot_offset']
            pre_reset_offset = config_push_world['pre_reset_offsets']
            pos_rel_box = np.array([reset_offset[0],reset_offset[1]+pre_reset_offset[1],reset_offset[2],1])
            pre_reset_pos = np.asarray(np.dot(pose,pos_rel_box)).ravel()[:3]

            reset_displacement = (self._box_reset_pos0 - reset_pos)
            reset_push = {'poses': [{'pos': pre_reset_pos, 'ori': reset_q}, {'pos': reset_pos, 'ori': reset_q}], 'push_action': reset_displacement, 'name' : 'reset_spot'}
            

            # pushes.append(reset_push)

            return pushes, pose, reset_push
        else:
            return [], None, None



class PushMachine(object):

    def __init__(self, robot_interface, sample_start_index=None):

        self._push_counter = 0
        self._box = BoxObject(robot_interface=robot_interface)
        self._robot = robot_interface

        self._states = {'RESET': 0, 'PUSH' : 1}
        self._state = self._states['RESET']

        self._record_sample = RecordSample(robot_interface=robot_interface, 
                                           task_interface=BoxObject(robot_interface=robot_interface),
                                           data_folder_path=config_push_world['data_folder_path'],
                                           data_name_prefix='sim_push_data',
                                           num_samples_per_file=500)


    def compute_next_state(self,idx):

        # Decide next state
        if idx == 0 and self._push_counter > 0 and self._state != self._states['RESET']:
            self._state = self._states['RESET']
        else:
            self._state = self._states['PUSH']

    def goto_next_state(self,idx,pushes, box_pose, reset_push):

        success = True

        # Take machine to next state
        if self._state == self._states['RESET']:
            # print "RESETING WITH NEW POSE"
            # #success = self.reset_box(reset_push)
            # os.system("spd-say 'Please reset the box, Much appreciated dear human'")
            # raw_input("Press enter to continue...")
            pass

        elif self._state == self._states['PUSH']:
            print "Moving to pre-push position ..."
                    
            # There might be a sequence of positions prior to a push action
            goals = self.pack_push_goals(pushes[idx])

            self._record_sample.record_once(task_action=pushes[idx])

            success = self.goto_goals(goals=goals, record=True, push = pushes[idx])

            goals.reverse()

            success = self.goto_goals(goals[1:])

            self._robot.untuck_arm()

            self._record_sample.record_once(task_action=None, task_status=success)

            if success:
                self._push_counter += 1
            
            idx = (idx+1)%(len(pushes))
        else:
            print "UNKNOWN STATE"


        return idx, success

    def pack_push_goals(self,push):

        goals = []
        for goal in push['poses']:
            goals.append(goal)

        push_action = push['push_action']

        goals.append({'pos': push_action, 'ori': None})


        return goals

    def goto_goals(self,goals, record=False, push = None):

        c = 0
        success = False
        for i in range(len(goals)-1):

            goal = goals[i]

            success = self.goto_pose(goal_pos=goal['pos'], goal_ori=None)

            if not success:
                return False

        # Push is always the last
        goal = goals[len(goals)-1]
        if record and push:
            print "Gonna push the box ..."
            # self._record_sample.start_record(push)
        
            success = self.goto_pose(goal_pos=goal['pos'], goal_ori=None)
            


        return success

    def on_shutdown(self):

        #this if for saving files in case keyboard interrupt happens
        self._record_sample.save_data_now()

    def run(self):

        push_finished = True
        pre_push_finished = True

        t = 0

        rate = rospy.Rate(10)

        idx = 0

        self._robot.untuck_arm()

        rospy.on_shutdown(self.on_shutdown)


        while not rospy.is_shutdown():# and not finished:

            self._robot._viewer.loop()

            pushes = None
            box_pose = None

            print "Moving to neutral position ..."
            
            pushes, box_pose, reset_push = self._box.get_pushes()
            
            self._box._last_pushes = pushes

            if pushes:

                self.compute_next_state(idx)

                if self._robot._state:

                    idx, success = self.goto_next_state(idx, pushes, box_pose, reset_push)

            rate.sleep()

        

    def goto_pose(self, goal_pos, goal_ori): 

        start_pos, start_ori = self._robot.get_ee_pose()

        if goal_ori is None:
             goal_ori = start_ori

        goal_ori    = quaternion.as_float_array(goal_ori)[0]

        js_pos      = np.zeros((7,1))

        js_pos[:3]  = goal_pos[:,None]

        js_pos[3:6] = goal_ori[1:][:,None]

        js_pos[6]   = goal_ori[0]

        success     = True

        if success:
            self._robot.set_qpos(np.vstack([self._robot._model.data.qpos[0:7], js_pos]))
        else:
            print "Couldnt find a solution"

        return success

    def reset_box(self,reset_push):
        success = True
        # There might be a sequence of positions prior to a push action
        for goal in reset_push['poses']:

            success = success and self.goto_pose(goal_pos=goal['pos'], goal_ori=None)


        ee_pos, _ = self._robot.get_ee_pose()

        success = success and self.goto_pose(goal_pos=ee_pos+reset_push['push_action'], goal_ori=None)

        return success


if __name__ == "__main__":
    
    rospy.init_node('poke_box', anonymous=True)

    robot_interface = MujocoRobot(xml_path=config_push_world['model_name'])

    viewer = MujocoViewer(mujoco_robot=robot_interface, width=config_push_world['image_width'], height=config_push_world['image_height'])

    viewer.configure(cam_pos=config_push_world['camera_pos'])

    robot_interface._configure(viewer=viewer, p_start_idx=7, p_end_idx=14, v_start_idx=6, v_end_idx=12)
    
    push_machine = PushMachine(robot_interface=robot_interface)

    push_machine.run()