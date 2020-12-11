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
        
        self.path_trajectory, self.path_pcd = self.check_data_path(data_dir)

        poses = genfromtxt(self.path_trajectory, delimiter=',')

        self.QUEUE_NUM = 10
        self.veldynes = []
        self.odoms = []
        num_frames = self.get_num_frames()

        index = 0

        if not os.path.exists(os.path.join(data_dir, 'imgs')):
            os.mkdir(os.path.join(data_dir, 'imgs'))
        time.sleep(1)
        
        for i in tqdm(range(index, index + num_frames)):
            file = os.path.join(data_dir, 'global/{0:06d}.pcd'.format(i))
            points = pcl.load(file)

            vels = self.update_velodyne_queue(points, poses[i][1::])

            self.save_vels(vels, file_name=os.path.join(data_dir, 'imgs/{0:06d}.png'.format(i)))

    def get_num_frames(self):
        num_trajectory = len(genfromtxt(self.path_trajectory, delimiter=','))
        num_pcd = len(os.listdir(self.path_pcd))
        num_frames = min(num_trajectory,num_pcd)
        print("Frames:", num_frames)

        return num_frames

    def check_data_path(self,path):
        file_list = os.listdir(path)

        # locate trajectory file
        path_trajectory = os.path.join(path,'trajectory.csv')
        if not os.path.exists(path_trajectory):
            print('Warning: Trajectory file not found! Try to find replacement.')
            path_trajectory = ''
            for file in file_list:
                if file.endswith('.csv'):
                    path_trajectory = os.path.join(path,file)
                    print("Set trajectory to '"+file+"'\n")
                    break
            if path_trajectory == '':
                raise Exception('Missing trajectory file')
        
        # locate pcd folder
        path_pcd = os.path.join(path,'pcd')
        if not os.path.exists(path_pcd):
            print('Warning: pcd folder not found! Try to find replacement.')
            path_pcd = ''
            folder_list = [s for s in file_list if os.path.isdir(os.path.join(path,s)) and not s.startswith('.')]
            for folder in folder_list:
                files = os.listdir(os.path.join(path,folder))
                if files[0].endswith('.pcd'):
                    path_pcd = os.path.join(path,folder)
                    print("Set pcd folder to '"+folder+"'\n")
                    break
            if path_pcd == '':
                raise Exception('Missing pcd file')

        return path_trajectory, path_pcd

    def transform_point_cloud(self, pc, T):

        pc = np.hstack([pc, np.ones([pc.shape[0], 1])])

        # transform to the local frame
        pc = np.matmul(T, pc.transpose())

        pc_transformed = pc.transpose()[:, :3]

        return pc_transformed

    def update_velodyne_queue(self, vel, pose):

        vel = np.asarray(vel)

        if len(self.veldynes) >= self.QUEUE_NUM:
            self.veldynes.pop(0)

        # transform the current scan from local to global frame (mapping)
        T = self.convet_quaternion_to_matrix(pose)
        #vel = self.transform_point_cloud(vel, T)
        self.veldynes.append(vel)

        # concate the velendy scans
        vels = self.veldynes[0]
        for i in range(1, len(self.veldynes)):
            vels = np.vstack([vels, self.veldynes[i]])

        # print(np.min(vels[:, 2]), np.max(vels[:, 2]))

        # transform the submap from global map frame to local lidar frame
        T2 = np.linalg.inv(T)
        vels2 = self.transform_point_cloud(vels, T2)

        # print(np.min(vels2[:, 2]), np.max(vels2[:, 2]))

        return vels2


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


    def save_vels(self, vels, file_name='tmp.png'):

        start_time = time.time()
        # 50, 50 res = 0.25 / 100, 100 res=0.5
        bird_view_img = lidar_projection.birds_eye_point_cloud(vels, side_range=(-100, 100), fwd_range=(-100, 100),
                                                               res=0.5, min_height=-8, max_height=20)
        # print("bird-view image shape: ", bird_view_img.shape)
        cv2.imwrite(file_name, bird_view_img)

        # print("[inference_node]: runing time = " + str(time.time() - start_time))


if __name__ == '__main__':
    dr = Self_Awareness('/dataset')
