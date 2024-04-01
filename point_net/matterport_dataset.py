import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class Matterport(Dataset):
    def __init__(self, root=None, split='train', npoints=4096, r_prob=0.25):
        self.root = root
        self.split = split.lower()  # use 'test' in order to bypass augmentations
        self.npoints = npoints  # use  None to sample all the points
        self.r_prob = r_prob  # probability of rotation

        # get all datapaths
        self.data_paths = []
        if root:
            self.data_paths = glob(os.path.join(root, split, '*.pth'), recursive=True)

    def __getitem__(self, idx):
        # read data from pth
        region_data = torch.load(self.data_paths[idx])
        points = region_data[0]  # xyz points
        colors = region_data[1]  # color points
        targets = region_data[2]  # integer categories

        # target_map = 20: ceiling
        #              0: wall
        #              1: floor
        #              All else: clutter

        # Need to map to S3DIS
        # target_map = 0: ceiling
        #              1: floor
        #              2: wall
        #              >2: clutter

        # Create a mapping dictionary
        target_map = {0: 2, 20: 0, 1: 1}

        # Initialize targets with a default value of 3
        mapped_targets = np.full_like(targets, fill_value=3)

        # Update values according to the mapping dictionary
        for original_value, mapped_value in target_map.items():
            mapped_targets[targets == original_value] = mapped_value

        targets = mapped_targets

        # down sample point cloud
        if self.npoints:
            points, colors, targets = self.downsample(points, colors, targets)

        # add Gaussian noise to point set if not testing
        if self.split != 'test':
            # add N(0, 1/100) noise
            points += np.random.normal(0., 0.01, points.shape)

            # add random rotation to the point cloud with probability
            if np.random.uniform(0, 1) > 1 - self.r_prob:
                points, colors = self.random_rotate(points, colors)

        # Normalize Point Cloud to (0, 1)
        points = self.normalize_points(points)
        colors = self.normalize_points(colors)

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        colors = torch.from_numpy(colors).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        return points, colors, targets

    def downsample(self, points, colors, targets):
        if len(points) > self.npoints:
            choice = np.random.choice(len(points), self.npoints, replace=False)
        else:
            # case when there are less points than the desired number
            choice = np.random.choice(len(points), self.npoints, replace=True)
        points = points[choice, :]
        colors = colors[choice, :]
        targets = targets[choice]

        return points, colors, targets

    @staticmethod
    def random_rotate(points, colors):
        ''' randomly rotates point cloud about vertical axis.
            Code is commented out to rotate about all axes
            '''
        # construct a randomly parameterized 3x3 rotation matrix
        phi = np.random.uniform(-np.pi, np.pi)
        theta = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)

        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]])

        rot_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]])

        rot_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]])

        # rot = np.matmul(rot_x, np.matmul(rot_y, rot_z))

        return np.matmul(points, rot_z), np.matmul(colors, rot_z)

    @staticmethod
    def normalize_points(points):
        ''' Perform min/max normalization on points
            Same as:
            (x - min(x))/(max(x) - min(x))
            '''
        points = points - points.min(axis=0)
        points /= points.max(axis=0)

        return points

    def __len__(self):
        return len(self.data_paths)