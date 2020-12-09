#!/usr/bin/python3.5

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2018-06-09 18:20:13
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-05-10 19:10:52



from toolbox import lidar_projection

from tqdm import tqdm
import numpy as np
import argparse
import os
import time
import pickle

import math
import cv2
import copy
from numpy import genfromtxt
import pcl


class Self_Awareness:

    def __init__(self, data_dir):


        poses = genfromtxt(os.path.join(data_dir, 'trajectory.csv'), delimiter=',')

        self.QUEUE_NUM = 5
        self.veldynes = []
        self.odoms = []

        index = 0
        index_total = poses.shape[0]

        if not os.path.exists(os.path.join(data_dir, 'imgs')):
            os.mkdir(os.path.join(data_dir, 'imgs'))

        for i in tqdm(range(index, index_total)):
            file = os.path.join(data_dir, 'global/{0:06d}.pcd'.format(i))
            points = pcl.load(file)

            vels = self.update_velodyne_queue(points, poses[i][1::])

            self.save_vels(vels, file_name=os.path.join(data_dir, 'imgs/{0:06d}.png'.format(i)))


    def get_relative_transform(self, q_old, q_t):

        M_old = self.convet_quaternion_to_matrix(q_old)
        M_t = self.convet_quaternion_to_matrix(q_t)

        t = np.array([q_old[0:3]-q_t[0:3]])

        R_old = M_old[0:3, 0:3]
        R_t = M_t[0:3, 0:3]
        deltaT = np.matmul(np.linalg.inv(R_t), R_old)

        deltaT = np.hstack([deltaT, t.transpose()])
        deltaT = np.vstack([deltaT, np.array([[0, 0, 0, 1]])])

        return deltaT

    def update_velodyne_queue(self, vel, pose):

        vel = np.asarray(vel)

        if len(self.veldynes) >= self.QUEUE_NUM:
            self.veldynes.pop(0)

        if vel.shape[0] > 0:
            self.veldynes.append(vel)

        # concate the velendy scans
        vels = self.veldynes[0]
        for i in range(1, len(self.veldynes)):
            vels = np.vstack([vels, self.veldynes[i]])

        vels = np.hstack([vels, np.ones([vels.shape[0], 1])])

        # transform to the local frame
        T = self.convet_quaternion_to_matrix(pose)
        vels = np.matmul(np.linalg.inv(T), vels.transpose())

        vels = vels.transpose()[:, :3]

        return vels

    def concat_scans(self):

        if len(self.veldynes) >= self.QUEUE_NUM:
            self.veldynes.pop(0)
            self.odoms.pop(0)

        # start with the current scan
        scans = np.asarray(self.veldynes[-1])
        curr_pose = self.odoms[-1]

        assert len(self.veldynes) == len(self.odoms) and len(self.veldynes) > 0

        # process the previous scans
        for i in range(len(self.veldynes)-1):
            scan = np.asarray(self.veldynes[i])

            deltaT = self.get_relative_transform(self.odoms[i], curr_pose)
            deltaT[0:3,0:3] = np.identity(3)

            scan = np.hstack([scan, np.ones([scan.shape[0], 1])])
            scan2 = np.matmul(deltaT, scan.transpose())

            scan2 = scan2.transpose()[:, :3]
            scans = np.vstack([scans, scan2])

        return scans

    def convet_quaternion_to_matrix(self, pose):

        [px, py, pz, q_x, q_y, _qz, q_w] = pose
        T = self.quaternion_matrix([q_x, q_y, _qz, q_w])
        T[0:3, 3] = [px, py, pz]

        return T

    def quaternion_matrix(self, quaternion):
        """Return homogeneous rotation matrix from quaternion.
        """
        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < np.finfo(float).eps * 4.0:
            return np.identity(4)
        q *= math.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)


    def do_conversion(self):

        start_time = time.time()

        # pc_np = np.asarray(self.veldynes[-1])

        pc_np = self.concat_scans()

        bird_view_img = lidar_projection.birds_eye_point_cloud(pc_np, side_range=(-50, 50), fwd_range=(-50, 50), res=0.25, min_height=-8, max_height=20)
        # print("bird-view image shape: ", bird_view_img.shape)
        cv2.imwrite(os.path.join("tmp.png"), bird_view_img)

        # print("[inference_node]: runing time = " + str(time.time() - start_time))

    def save_vels(self, vels, file_name='tmp.png'):

        start_time = time.time()
        bird_view_img = lidar_projection.birds_eye_point_cloud(vels, side_range=(-50, 50), fwd_range=(-50, 50),
                                                               res=0.25, min_height=-8, max_height=20)
        # print("bird-view image shape: ", bird_view_img.shape)
        cv2.imwrite(file_name, bird_view_img)

        # print("[inference_node]: runing time = " + str(time.time() - start_time))


if __name__ == '__main__':
    dr = Self_Awareness('/dataset')
