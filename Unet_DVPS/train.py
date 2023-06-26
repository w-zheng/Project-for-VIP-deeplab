from KITTI_dataloader import SemKittiDataset 
from model import Model
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sys
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt





@dataclass
class SemKITTI_Class:
    name: str
    ID: int
    hasInstances: bool
    color: Tuple[int, int, int]

# List of classes that we want to detect in the input
classes = {
    0   : SemKITTI_Class('car'               ,  0 , True , (  0,   0, 255)),
    1   : SemKITTI_Class('bicycle'           ,  1 , True , (245, 150, 100)),
    2   : SemKITTI_Class('motorcycle'        ,  2 , True , (245, 230, 100)),
    3   : SemKITTI_Class('truck'             ,  3 , True , (250,  80, 100)),
    4   : SemKITTI_Class('other-vehicle'     ,  4 , True , (150,  60,  30)),
    5   : SemKITTI_Class('person'            ,  5 , True , (111,  74,   0)),
    6   : SemKITTI_Class('bicyclist'         ,  6 , True , ( 81,   0,  81)),
    7   : SemKITTI_Class('motorcyclist'      ,  7 , True , (128,  64, 128)),
    8   : SemKITTI_Class('road'              ,  8 , False, (244,  35, 232)),
    9   : SemKITTI_Class('parking'           ,  9 , False, (250, 170, 160)),
    10  : SemKITTI_Class('sidewalk'          , 10 , False, (230, 150, 140)),
    11  : SemKITTI_Class('other-ground'      , 11 , False, ( 70,  70,  70)),
    12  : SemKITTI_Class('building'          , 12 , False, (102, 102, 156)),
    13  : SemKITTI_Class('fence'             , 13 , False, (190, 153, 153)),
    14  : SemKITTI_Class('vegetation'        , 14 , False, (180, 165, 180)),
    15  : SemKITTI_Class('trunk'             , 15 , False, (150, 100, 100)),
    16  : SemKITTI_Class('terrain'           , 16 , False, (150, 120,  90)),
    17  : SemKITTI_Class('pole'              , 17 , False, (153, 153, 153)),
    18  : SemKITTI_Class('traffic-sign'      , 18 , False, ( 50, 120, 255)),
    255 : SemKITTI_Class('unlabeled'        , 255 , False, (  0,   0,   0)),
}


dataset_train = SemKittiDataset(dir_input="./DATALOC/semkitti-dvps/video_sequence/train", classes=classes)
dataset_val = SemKittiDataset(dir_input="./DATALOC/semkitti-dvps/video_sequence/val", classes=classes)
ds_split = {
    "train": dataset_train,
    "val": dataset_val,
    "test": []
}





def compute_iou(output: torch.Tensor, truths: torch.Tensor) -> float:
    # split output tensor and put it on cpu
    output = output.detach().cpu() 
    truths = truths.detach().cpu()
    batch_num = output.size()[0]
    class_num = output.size()[1]
    IOU = []
    
    # Initialize intersection and union lists for each class
    union = [0] * class_num
    
    # Calculate intersection and union for each class
    for i in range(batch_num):
        intersection = [0] * class_num
        class_id = torch.max(output[i],dim = 0).indices # output[i].shape = (num_class, height, width), using indices to have (572, 572)
        for j in range(class_num):
            pred_mask = (class_id == j)
            true_mask = (truths[i] == j)
            intersection[j] = (pred_mask & true_mask).sum()
            union[j] = (pred_mask | true_mask).sum()

        # in case we have zero on the denominator, use max to invoid a zero
        IOU.append(sum(intersection) / max(sum(union), 1e-5))
    iou = sum(IOU)/batch_num

    return iou


def absRel(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.detach().cpu() 
    y_true = y_true.detach().cpu()

    abs_diff = torch.abs(y_pred - y_true)
    rel_diff = abs_diff / y_true
    avg_rel_diff = torch.mean(rel_diff)
    return avg_rel_diff.item()

from torch import nn, optim
from torch.utils.data import DataLoader
import sys

class Trainer:
    def __init__(self, model: nn.Module, ds_split: Dict[str,SemKittiDataset], lr):
        # Choose a device to run training on. Ideally, you have a GPU available to accelerate the training process.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move the model onto the target device
        self.model = model.to(self.device)

        # Store the dataset split separately
        self.ds_split = ds_split

        # use Adam to optimize and Cross Entropy Loss to train
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.criterion_depth = nn.MSELoss()

        # test if optimizer and loss works
        assert self.optimizer is not None, "You have not defined an optimizer"
        assert self.criterion is not None, "You have not defined a loss for semantic segmentation"
        assert self.criterion_depth is not None, "You have not defined a loss for depth estimation"

    def train_epoch(self, dl:DataLoader):
        # Put the model in training mode
        self.model.train()

        # Store each step's accuracy and loss for this epoch
        epoch_metrics = {
            "loss_seg": [],
            "loss_depth": [],
            "loss": [],
            "accuracy_seg": [],
            "absRel": []
        }

        # Create a progress bar using TQDM
        sys.stdout.flush()
        with tqdm(total=len(self.ds_split["train"]), desc=f'Training') as pbar:
            # Iterate over the training dataset
            for inputs, truths in dl:
                # save for multi-task
                truth_depth = truths[0]
                truth_segment = truths[1].squeeze(dim=1)
                truth_instance = truths[2].squeeze(dim=1)

                # Zero the gradients from the previous step
                self.optimizer.zero_grad()

                # Move the inputs and truths to the target device
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                inputs.required_grad = True  # Fix for older PyTorch versions

                mask_depth = truth_depth > 0
                # mask_segment = truth_segment < 255
                mask_instance = truth_instance > 0
                
                # truths = torch.tensor(truths)
                truth_segment = truth_segment.to(device=self.device, dtype=torch.long)
                truth_depth = truth_depth.to(device=self.device, dtype=torch.float32)

                # Run model on the inputs
                output_seg, output_depth = self.model(inputs) 

                # Perform backpropagation
                loss_seg = self.criterion(output_seg, truth_segment)
                loss_depth = self.criterion_depth(output_depth[mask_depth]/torch.max(truth_depth[mask_depth]), truth_depth[mask_depth]/torch.max(truth_depth[mask_depth]))
                # print(truth_depth[mask])
                # weighted loss
                loss = loss_depth+loss_seg
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                self.optimizer.step()

                # Store the metrics of this step
                step_metrics = {
                    'loss_seg': loss_seg.item(),
                    'loss_depth': loss_depth.item(),
                    'loss': loss.item(),
                    'accuracy_seg': compute_iou(output_seg, truth_segment),
                    "absRel": absRel(output_depth[mask_depth], truth_depth[mask_depth])
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(list(inputs.shape)[0])

                # Add to epoch's metrics
                for k,v in step_metrics.items():
                    epoch_metrics[k].append(v)

        sys.stdout.flush()

        # Return metrics
        return epoch_metrics

    def val_epoch(self, dl:DataLoader):
        # Put the model in evaluation mode
        self.model.eval()

        # Store the total loss and accuracy over the epoch
        amount = 0
        total_loss_seg=0
        total_loss_depth=0
        total_loss = 0
        total_accuracy = 0
        total_absRel = 0

        # Create a progress bar using TQDM
        sys.stdout.flush()
        with torch.no_grad(), tqdm(total=len(self.ds_split["val"]), desc=f'Validation') as pbar:
            # Iterate over the validation dataloader
            for inputs, truths in dl:
                # save for multi-task
                truth_depth = truths[0]
                truth_segment = truths[1].squeeze(dim=1)
                truth_instance = truths[2].squeeze(dim=1)

                # Move the inputs and truths to the target device
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                
                inputs.required_grad = True  # Fix for older PyTorch versions

                truth_segment = truth_segment.to(device=self.device, dtype=torch.long)
                truth_depth = truth_depth.to(device=self.device, dtype=torch.float32)

                mask_depth = truth_depth > 0
                # mask_segment = truth_segment < 255
                mask_instance = truth_instance > 0

                # Run model on the inputs
                output_seg, output_depth = self.model(inputs) 

                # Perform backpropagation
                loss_seg = self.criterion(output_seg, truth_segment)
                loss_depth = self.criterion_depth(output_depth[mask_depth]/torch.max(truth_depth[mask_depth]), truth_depth[mask_depth]/torch.max(truth_depth[mask_depth]))

                # weighted loss
                loss = loss_depth + loss_seg

                # Store the metrics of this step
                step_metrics = {
                    'loss_seg': loss_seg.item(),
                    'loss_depth': loss_depth.item(),
                    'loss': loss.item(),
                    'accuracy_seg': compute_iou(output_seg, truth_segment),
                    "absRel": absRel(output_depth[mask_depth], truth_depth[mask_depth])
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(list(inputs.shape)[0])

                amount += 1
                total_loss_depth += step_metrics["loss_depth"]
                total_loss_seg += step_metrics["loss_seg"]
                total_loss += step_metrics["loss"]
                total_accuracy += step_metrics["accuracy_seg"]
                total_absRel += step_metrics["absRel"]
        sys.stdout.flush()

        # Print mean of metrics
        total_loss /= amount
        total_accuracy /= amount
        # total_absRel /= amount
        print(f'Validation loss is {total_loss/amount}, validation accuracy is {total_accuracy}, validation absRel error is {total_absRel/amount}')

        # Return mean loss and accuracy
        return {
            'loss_seg': [total_loss_depth],
            'loss_depth': [total_loss_seg],
            'loss': [total_loss],
            'accuracy_seg': [total_accuracy],
            "absRel": [total_absRel]
        }

    def fit(self, epochs: int, batch_size:int):
        # Initialize Dataloaders for the `train` and `val` splits of the dataset. 
        # A Dataloader loads a batch of samples from the each dataset split and concatenates these samples into a batch.
        dl_train = DataLoader(ds_split["train"], batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(ds_split["val"], batch_size=batch_size, drop_last=True)

        # # use subset for test
        # # Determine the number of samples to use for training
        # num_samples = len(ds_split["train"])
        # num_val = len(ds_split["val"])
        # num_train_samples = num_samples // 1
        # num_val_samples = num_val // 1

        # # Create a subset of the training dataset with 1/100 samples
        # train_subset = torch.utils.data.Subset(ds_split["train"], list(range(num_train_samples)))
        # val_subset = torch.utils.data.Subset(ds_split["train"], list(range(num_val_samples)))

        # # Create the DataLoader for the training subset
        # dl_train_subset = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        # dl_val_subset = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

        # Store metrics of the training process (plot this to gain insight)
        df_train = pd.DataFrame()
        df_val = pd.DataFrame()

        # Train the model for the provided amount of epochs
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}')
            metrics_train = self.train_epoch(dl_train)
            # print(metrics_train)
            # df_train = df_train.append(pd.DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss"]))], **metrics_train}), ignore_index=True)
            df_train = pd.concat([df_train,pd.DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss_seg"]))], **metrics_train})], ignore_index=True)
            metrics_val = self.val_epoch(dl_val)
            # df_val = df_val.append(pd.DataFrame({'epoch': [epoch], **metrics_val}), ignore_index=True)
            df_val = pd.concat([df_val,pd.DataFrame({'epoch': [epoch for _ in range(len(metrics_val["loss_seg"]))], **metrics_val})], ignore_index=True)

        # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.
        return df_train, df_val




# create folder
folder_name = 'unet_seg_depth'
p_model = (Path()/'Train'/folder_name)
p_model.mkdir(parents=True, exist_ok=True)

# free memory
torch.cuda.empty_cache()

# Train the passthrough network
print("Testing training process...")
model = Model()

# torch.cuda.empty_cache()
trainer = Trainer(model, ds_split,lr = 1e-4)
train_loss, validation_loss = trainer.fit(epochs=30, batch_size=16)

# Draw a random sample
input, truth = random.choice(ds_split["val"])

# Push through our network
model = model.cpu()
output = model(input.unsqueeze(0))

# save model in the folder
model_name = f"model.pth"
torch.save(model.state_dict(),(p_model/model_name))

# save dataframe file
train_loss.to_csv((p_model/f"train_loss.csv"))
validation_loss.to_csv((p_model/f"validation_loss.csv"))


train_acc =[]
vad_acc=[]
painted = 'loss_seg'
for i in range(1,train_loss['epoch'].max()):
    train_acc.append(train_loss[train_loss['epoch']==i][painted].mean(0))
    vad_acc.append(validation_loss[validation_loss['epoch']==i][painted].mean(0))

plt.plot(train_acc,'b-')
plt.plot(vad_acc,'r--')

plt.title(painted)
plt.ylabel(painted)
plt.xlabel('epoch')
plt.legend(['train', 'val'])
pic_name = f"Train_validation_loss.png"
plt.savefig((p_model/pic_name))



train_acc1 =[]
vad_acc1=[]
painted = 'accuracy_seg'
for i in range(1,train_loss['epoch'].max()):
    train_acc1.append(train_loss[train_loss['epoch']==i][painted].mean(0))
    vad_acc1.append(validation_loss[validation_loss['epoch']==i][painted].mean(0))

plt.plot(train_acc1,'b-')
plt.plot(vad_acc1,'r--')

plt.title(painted)
plt.ylabel(painted)
plt.xlabel('epoch')
plt.legend(['train', 'val'])
pic_name = f"Train_validation_accuracy.png"
plt.savefig((p_model/pic_name))
