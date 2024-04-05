import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataset import CombinedDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassMatthewsCorrCoef

from randla_net import RandLANet

# dataset
S3DIS_ROOT = r'/home/asakhare/data/datasets/stanford/Stanford3dDataset_v1.2_Reduced_Aligned_Version'
MATTERPORT_ROOT = r'/home/asakhare/data/datasets/matterport/matterport_3d'
SCANNET_ROOT = r'/home/asakhare/data/datasets/scannet/scannet_3d'
WEIGHTS_PATH = r'./trained_models/s3dis/randla_net/seg_model_54.pth'

# feature selection hyperparameters
NUM_TRAIN_POINTS = 200000 # train/valid points
NUM_TEST_POINTS = 100000
BATCH_SIZE = 4
EPOCHS = 200
LR = 0.00001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CATEGORIES = {'ceiling'  : 0,
              'floor'    : 1,
              'wall'     : 2,
              'window'   : 3,
              'door'     : 4,
              'clutter'  : 5
              }

NUM_CLASSES = len(CATEGORIES)

COLOR_MAP = {0 : (255, 0, 0), # ceiling - red
             1: (0, 255, 0), # floor - lime
             2: (0, 0, 255), # wall - blue
             3: (0, 255, 255), # window - aqua
             4: (255, 255, 0), # door - yellow
             5: (0, 0, 0) # clutter - black
             }


def compute_iou(targets, predictions):

    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    intersection = torch.sum(predictions == targets) # true positives
    union = len(predictions) + len(targets) - intersection

    return intersection / union

# Initialize wandb
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)
wandb.init(project='PointNet', entity='asakhare')

# get datasets
train_dataset = CombinedDataset(s3dis_root=S3DIS_ROOT,
                                matterport_root=None,
                                scannet_root=None,
                                s3dis_area_nums='1-4',
                                npoints=NUM_TRAIN_POINTS,
                                r_prob=0.25
                                )
valid_dataset = CombinedDataset(s3dis_root=S3DIS_ROOT,
                                matterport_root=None,
                                scannet_root=None,
                                s3dis_area_nums='5',
                                npoints=NUM_TRAIN_POINTS,
                                r_prob=0.
                                )
test_dataset = CombinedDataset(s3dis_root=S3DIS_ROOT,
                               matterport_root=MATTERPORT_ROOT,
                               scannet_root=SCANNET_ROOT,
                               s3dis_area_nums='6',
                               split='test',
                               npoints=NUM_TEST_POINTS
                               )

# get dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

seg_model = RandLANet(d_in=6,
                      num_classes=NUM_CLASSES,
                      num_neighbors=16,
                      decimation=4,
                      device=DEVICE
                      )
seg_model.load_state_dict(torch.load(WEIGHTS_PATH))

points, colors, targets = next(iter(train_dataloader))
out, _, _ = seg_model(torch.cat((points.transpose(2, 1), colors.transpose(2,1)), dim=1))
print(f'Seg shape: {out.shape}')

train_targets = []
for (_, _, targets) in train_dataloader:
    train_targets += targets.reshape(-1).numpy().tolist()

train_targets = np.array(train_targets)
class_bins = np.bincount(train_targets)
total_count = np.sum(class_bins)  # Total count of all categories
percentages = class_bins / total_count  # Calculating percentages
weights = [1 / p for p in percentages]
max_weight = max(weights)
alpha = [w / max_weight for w in weights]

optimizer = optim.Adam(seg_model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(weight=alpha).to(DEVICE)

seg_model = seg_model.to(DEVICE)
mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)

# store best validation iou
best_iou = 0.4
best_mcc = 0.4

# lists to store metrics
train_loss = []
train_accuracy = []
train_mcc = []
train_iou = []
valid_loss = []
valid_accuracy = []
valid_mcc = []
valid_iou = []

# stuff for training
num_train_batch = int(np.ceil(len(train_dataset) / BATCH_SIZE))
num_valid_batch = int(np.ceil(len(valid_dataset) / BATCH_SIZE))

for epoch in range(1, EPOCHS + 1):
    # place model in training mode
    seg_model = seg_model.train()
    _train_loss = []
    _train_accuracy = []
    _train_mcc = []
    _train_iou = []
    for i, (points, colors, targets) in enumerate(train_dataloader, 0):

        points = points.transpose(2, 1).to(DEVICE)
        colors = colors.transpose(2, 1).to(DEVICE)
        points = torch.cat((points, colors), dim=1)
        targets = targets.squeeze().to(DEVICE)

        # zero gradients
        optimizer.zero_grad()

        # get predicted class logits
        preds = seg_model(points)

        pred_choice_logp = torch.distributions.utils.probs_to_logits(preds, is_binary=False)
        loss = criterion(pred_choice_logp, targets)

        loss.backward()
        optimizer.step()

        # get metrics
        pred_targets = torch.max(preds, dim=-2).indices
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct / float(BATCH_SIZE * NUM_TRAIN_POINTS)
        # mcc = mcc_metric(preds.transpose(2, 1), targets)
        iou = compute_iou(targets, pred_choice)

        # update epoch loss and accuracy
        _train_loss.append(loss.item())
        _train_accuracy.append(accuracy)
        _train_mcc.append(mcc.item())
        _train_iou.append(iou.item())

        if i % 100 == 0:
            print(f'\t [{epoch}: {i}/{num_train_batch}] ' \
                  + f'train loss: {loss.item():.4f} ' \
                  + f'accuracy: {accuracy:.4f} ' \
                  + f'mcc: {mcc:.4f} ' \
                  + f'iou: {iou:.4f}')

    train_loss.append(np.mean(_train_loss))
    train_accuracy.append(np.mean(_train_accuracy))
    train_mcc.append(np.mean(_train_mcc))
    train_iou.append(np.mean(_train_iou))

    print(f'Epoch: {epoch} - Train Loss: {train_loss[-1]:.4f} ' \
          + f'- Train Accuracy: {train_accuracy[-1]:.4f} ' \
          + f'- Train MCC: {train_mcc[-1]:.4f} ' \
          + f'- Train IOU: {train_iou[-1]:.4f}')

    # pause to cool down
    time.sleep(1)

    # get test results after each epoch
    with torch.no_grad():

        # place model in evaluation mode
        seg_model = seg_model.eval()

        _valid_loss = []
        _valid_accuracy = []
        _valid_mcc = []
        _valid_iou = []
        for i, (points, colors, targets) in enumerate(valid_dataloader, 0):

            points = points.transpose(2, 1).to(DEVICE)
            colors = colors.transpose(2, 1).to(DEVICE)
            points = torch.cat((points, colors), dim=1)
            targets = targets.squeeze().to(DEVICE)

            preds = seg_model(points)

            pred_choice_logp = torch.distributions.utils.probs_to_logits(preds, is_binary=False)
            loss = criterion(pred_choice_logp, targets)

            # get metrics
            pred_choice = torch.max(preds, dim=-2).indices
            correct = pred_choice.eq(targets.data).cpu().sum()
            accuracy = correct / float(BATCH_SIZE * NUM_TRAIN_POINTS)
            mcc = mcc_metric(preds.transpose(2, 1), targets)
            iou = compute_iou(targets, pred_choice)

            # update epoch loss and accuracy
            _valid_loss.append(loss.item())
            _valid_accuracy.append(accuracy)
            _valid_mcc.append(mcc.item())
            _valid_iou.append(iou.item())

            if i % 100 == 0:
                print(f'\t [{epoch}: {i}/{num_valid_batch}] ' \
                      + f'valid loss: {loss.item():.4f} ' \
                      + f'accuracy: {accuracy:.4f} '
                      + f'mcc: {mcc:.4f} ' \
                      + f'iou: {iou:.4f}')

        valid_loss.append(np.mean(_valid_loss))
        valid_accuracy.append(np.mean(_valid_accuracy))
        # valid_mcc.append(np.mean(_valid_mcc))
        valid_iou.append(np.mean(_valid_iou))
        print(f'Epoch: {epoch} - Valid Loss: {valid_loss[-1]:.4f} ' \
              + f'- Valid Accuracy: {valid_accuracy[-1]:.4f} ' \
              + f'- Valid MCC: {valid_mcc[-1]:.4f} ' \
              + f'- Valid IOU: {valid_iou[-1]:.4f}')

        # pause to cool down
        time.sleep(1)

    # save best models
    if valid_iou[-1] >= best_iou:
        best_iou = valid_iou[-1]
        torch.save(seg_model.state_dict(), f'trained_models/custom/seg_model_{epoch}.pth')

    # Log metrics after each epoch
    wandb.log({'train_loss': train_loss[-1],
               'train_accuracy': train_accuracy[-1],
               'train_mcc': train_mcc[-1],
               'train_iou': train_iou[-1]}
              )

    wandb.log({'valid_loss': valid_loss[-1],
               'valid_accuracy': valid_accuracy[-1],
               'valid_mcc': valid_mcc[-1],
               'valid_iou': valid_iou[-1]}
              )
