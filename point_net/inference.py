import numpy as np
import open3d as o3d
import torch
from point_net import PointNetSegHead
from pathlib import Path

# Read the data
PCD_FILEPATH = r'C:\Users\AshwinSakhare\MyDrive\GitHub\zData\koda_wayfinding\data\point_cloud_store_.1.ply'
WEIGHTS_FILEPATH = r'C:\Users\AshwinSakhare\MyDrive\GitHub\zData\3D\point_net\trained_models\custom\seg_model_162.pth'
PCD_FILEPATH = r'C:\Users\AshwinSakhare\Desktop\S3DIS\Area_6\conferenceRoom_1.pth'
# PCD_FILEPATH = '/home/asakhare/data/datasets/stanford/Stanford3dDataset_v1.2_Reduced_Aligned_Version/Area_1/conferenceRoom_1.pth'
# WEIGHTS_FILEPATH = '/home/asakhare/github/zData/3D/point_net/trained_models/custom/seg_model_162.pth'
NUM_POINTS = 250000
COLOR_MAP = {0 : (255, 0, 0), # ceiling - red
             1: (0, 255, 0), # floor - lime
             2: (0, 0, 255), # wall - blue
             3: (0, 0, 0) # clutter - black
             }

v_map_colors = np.vectorize(lambda x : COLOR_MAP[x])

seg_model = PointNetSegHead(num_points=NUM_POINTS, m=4)
weights = torch.load(WEIGHTS_FILEPATH)
seg_model.load_state_dict(weights)
seg_model = seg_model.to('cuda')
seg_model.eval()

if Path(PCD_FILEPATH).suffix == '.ply':
    pcd = o3d.io.read_point_cloud(PCD_FILEPATH)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

else:
    pcd = torch.load(PCD_FILEPATH)
    points = pcd[:, :3]  # xyz points
    points = np.asarray(points)
    colors = pcd[:, 3:6]  # color points
    colors = np.asarray(colors)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors / 255)
o3d.visualization.draw_geometries([pcd])


points = points - points.min(axis=0)
points /= points.max(axis=0)
points = torch.tensor(points, dtype=torch.float32)


colors = colors - colors.min(axis=0)
colors /= colors.max(axis=0)
colors = torch.tensor(colors, dtype=torch.float32)

input = torch.cat((points, colors), dim=1)
input = input.unsqueeze(0)
input = input.transpose(1,2)
input = input.to('cuda')

choice = np.random.choice(input.shape[2], NUM_POINTS, replace=False)

input = input[:, :, choice]

output, _, _ = seg_model(input)

pred_labels = torch.softmax(output, dim=2).argmax(dim=2)
pred_labels = pred_labels.reshape(-1).cpu()

pcd = o3d.geometry.PointCloud()

points = input[0, 0:3, :].cpu().numpy()
points = np.transpose(points, (1,0))
pcd.points = o3d.utility.Vector3dVector(points)

colors = np.vstack(v_map_colors(pred_labels)).T/255
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])