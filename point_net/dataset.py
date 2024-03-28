from torch.utils.data import Dataset

from matterport_dataset import Matterport
from s3dis_dataset import S3DIS


class CombinedDataset(Dataset):
    def __init__(self, s3dis_root, matterport_root, s3dis_area_nums, split='train', npoints=4096, r_prob=0.25):
        self.s3dis_dataset = S3DIS(root=s3dis_root, area_nums=s3dis_area_nums, split=split, npoints=npoints, r_prob=r_prob)
        self.matterport_dataset = Matterport(root=matterport_root, split=split, npoints=npoints, r_prob=r_prob)

        # Length of combined dataset
        self.length = len(self.s3dis_dataset) + len(self.matterport_dataset)

    def __getitem__(self, idx):
        # Randomly select from either S3DIS or Matterport datasets
        if idx < len(self.s3dis_dataset):
            return self.s3dis_dataset[idx]
        else:
            # Adjust index for Matterport dataset
            idx -= len(self.s3dis_dataset)
            return self.matterport_dataset[idx]

    def __len__(self):
        return self.length
