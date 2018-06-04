#  handEye - performs hand/eye calibration
#  
#      gHc = handEye(bHg, wHc)
#  
#      bHg - pose of gripper relative to the robot base..
#            (Gripper center is at: g0 = Hbg * [0;0;0;1] )
#            Matrix dimensions are 4x4xM, where M is ..
#            .. number of camera positions. 
#            Algorithm gives a non-singular solution when ..
#            .. at least 3 positions are given
#            Hbg(:,:,i) is i-th homogeneous transformation matrix
#      wHc - pose of camera relative to the world ..      
#            (relative to the calibration block)
#            Dimension: size(Hwc) = size(Hbg)
#      gHc - 4x4 homogeneous transformation from gripper to camera      
#            , that is the camera position relative to the gripper.
#            Focal point of the camera is positioned, ..
#            .. relative to the gripper, at
#                  f = gHc*[0;0;0;1];
#            
#  References: R.Tsai, R.K.Lenz "A new Technique for Fully Autonomous 
#            and Efficient 3D Robotics Hand/Eye calibration", IEEE 
#            trans. on robotics and Automaion, Vol.5, No.3, June 1989
# 
#  Notation: wHc - pose of camera frame (c) in the world (w) coordinate system
#                  .. If a point coordinates in camera frame (cP) are known
#                  ..     wP = wHc * cP
#                  .. we get the point coordinates (wP) in world coord.sys.
#                  .. Also refered to as transformation from camera to world
# 

import numpy as np
from aml_math.quaternion_utils import rot2quat2 as rot2quat, quat2rot, transl, skew


class HandEyeCalib(object):
    def __init__(self):

        self._ee_poses = []
        self._camera_poses = []

    def add_measurement(self, ee_pose, cam_pose):

        self._add_ee_pose(ee_pose)
        self._add_camera_pose(cam_pose)

    def _add_ee_pose(self, pose):

        """ Add end-effector pose w.r.t to robot base frame """

        self._ee_poses.append(pose)

    def _add_camera_pose(self, pose):

        """ Add camera pose w.r.t world frame """

        self._camera_poses.append(pose)

    def calibrate(self):

        M = len(self._ee_poses)

        Hg = np.zeros((M, 4, 4))
        wHc = np.zeros((M, 4, 4))

        for i in range(M):
            Hg[i, :, :] = self._ee_poses[i]
            wHc[i, :, :] = self._camera_poses[i]


        K = (M * M - M) / 2  # Number of unique camera position pairs
        A = np.zeros((3 * K, 3))  # will store: skew(Pgij+Pcij)
        B = np.zeros((3 * K, 1))  # will store: Pcij - Pgij
        k = 0
        #
        # # Now convert from wHc notation to Hc notation used in Tsai paper.
        # # Hc = cHw = inv(wHc); We do it in a loop because wHc is given, not cHw
        Hc = np.zeros((M, 4, 4))
        #
        for i in range(M):
            Hc[i, :, :] = np.linalg.inv(wHc[i, :, :])  # wHc[i, :, :]

        for i in range(M):
            for j in range(i + 1, M):
                Hgij = np.dot(np.linalg.inv(Hg[j, :, :]),
                              Hg[i, :, :])  #  Transformation from i-th to j-th gripper pose

                # print "LOOOK",
                Pgij = 2 * rot2quat(Hgij[:3, :3])  #  ... and the corresponding quaternion

                Hcij = np.dot(Hc[j, :, :], np.linalg.inv(Hc[i, :, :]))  #  Transformation from i-th to j-th camera pose

                Pcij = 2 * rot2quat(Hcij[:3, :3])  #  ... and the corresponding quaternion

                #  Form linear system of equations
                sk = skew(Pgij + Pcij)

                A[3 * k:3 * k + 3, :3] = sk.copy()  #  left-hand side

                # print "What: ",B[(3*k-3)+np.array(range(3))].shape, (Pcij - Pgij).shape
                B[3 * k:3 * k + 3] = Pcij - Pgij  #  right-hand side
                k = k + 1

        # #  Rotation from camera to gripper is obtained from the set of equations:
        #   #     skew(Pgij+Pcij) * Pcg_ = Pcij - Pgij
        #   #  Gripper with camera is first moved to M different poses, then the gripper
        #   #  .. and camera poses are obtained for all poses. The above equation uses
        #   #  .. invariances present between each pair of i-th and j-th pose.
        #
        #
        Pcg_, _, _, _ = np.linalg.lstsq(A, B, rcond=-1)  #  Solve the equation A*Pcg_ = B
        #
        # #   Obtained non-unit quaternin is scaled back to unit value that
        # #   .. designates camera-gripper rotation
        #
        #
        Pcg = 2 * Pcg_ / np.sqrt(1.0 + np.dot(Pcg_.T, Pcg_))
        #
        Rcg = quat2rot(Pcg / 2)  #  Rotation matrix

        #
        #
        # #  Calculate translational component
        k = 0

        A = np.zeros((3 * K, 3))  # will store: skew(Pgij+Pcij)
        B = np.zeros((3 * K, 1))  # will store: Pcij - Pgij
        for i in range(0, M):
            for j in range(i + 1, M):
                Hgij = np.dot(np.linalg.inv(Hg[j, :, :]),
                              Hg[i, :, :])  # Transformation from i-th to j-th gripper pose
                Hcij = np.dot(Hc[j, :, :], np.linalg.inv(Hc[i, :, :]))  # Transformation from i-th to j-th camera pose

                A[3 * k:3 * k + 3, :3] = Hgij[:3, :3] - np.eye(3)  # left-hand side

                B[3 * k:3 * k + 3] = np.dot(Rcg[:3, :3], Hcij[:3, 3].reshape(3, 1)) - Hgij[:3, 3].reshape(3,1)  # right-hand side

                k = k + 1

        Tcg, _, _, _ = np.linalg.lstsq(A, B, rcond=-1)

        gHc = np.dot(transl(Tcg), Rcg)

        return gHc
