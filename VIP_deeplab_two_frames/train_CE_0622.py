# import

from KITTI_dataloader import SemKittiDataset
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from IPython.display import display, HTML
from io import BytesIO
from base64 import b64encode
from Model import Model
from heatmap_generate import _generate_gt_center_and_offset
from ins_decoder import FinalInstanceDecoder
from DeepLabCE import DeepLabCE
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset
from torch.utils.data import RandomSampler, SubsetRandomSampler
# dataloader
from Depth_loss import DepthLoss


@dataclass
class SemKITTI_Class:
    name: str
    ID: int
    hasInstances: bool
    color: Tuple[int, int, int]


# List of classes that we want to detect in the input
classes = {
    0: SemKITTI_Class("car", 0, True, (0, 0, 255)),
    1: SemKITTI_Class("bicycle", 1, True, (245, 150, 100)),
    2: SemKITTI_Class("motorcycle", 2, True, (245, 230, 100)),
    3: SemKITTI_Class("truck", 3, True, (250, 80, 100)),
    4: SemKITTI_Class("other-vehicle", 4, True, (150, 60, 30)),
    5: SemKITTI_Class("person", 5, True, (111, 74, 0)),
    6: SemKITTI_Class("bicyclist", 6, True, (81, 0, 81)),
    7: SemKITTI_Class("motorcyclist", 7, True, (128, 64, 128)),
    8: SemKITTI_Class("road", 8, False, (244, 35, 232)),
    9: SemKITTI_Class("parking", 9, False, (250, 170, 160)),
    10: SemKITTI_Class("sidewalk", 10, False, (230, 150, 140)),
    11: SemKITTI_Class("other-ground", 11, False, (70, 70, 70)),
    12: SemKITTI_Class("building", 12, False, (102, 102, 156)),
    13: SemKITTI_Class("fence", 13, False, (190, 153, 153)),
    14: SemKITTI_Class("vegetation", 14, False, (180, 165, 180)),
    15: SemKITTI_Class("trunk", 15, False, (150, 100, 100)),
    16: SemKITTI_Class("terrain", 16, False, (150, 120, 90)),
    17: SemKITTI_Class("pole", 17, False, (153, 153, 153)),
    18: SemKITTI_Class("traffic-sign", 18, False, (50, 120, 255)),
    255: SemKITTI_Class("unlabeled", 255, False, (0, 0, 0)),
}


dataset_train = SemKittiDataset(
    dir_input="./DATALOC/semkitti-dvps/video_sequence/train", classes=classes
)
dataset_val = SemKittiDataset(
    dir_input="./DATALOC/semkitti-dvps/video_sequence/val", classes=classes
)



# ds_split = {"train": dataset_train, "val": dataset_val, "test": []}

dataset_train_next = SemKittiDataset(dir_input="./DATALOC/semkitti-dvps/video_sequence/train", classes=classes)
dataset_train_next.items = dataset_train.items[1:]

dataset_train_previous = SemKittiDataset(dir_input="./DATALOC/semkitti-dvps/video_sequence/train", classes=classes)
dataset_train_previous.items = dataset_train.items[:-1]

dataset_val_next = SemKittiDataset(dir_input="./DATALOC/semkitti-dvps/video_sequence/val", classes=classes)
dataset_val_next.items = dataset_val.items[1:]

dataset_val_previous = SemKittiDataset(dir_input="./DATALOC/semkitti-dvps/video_sequence/val", classes=classes)
dataset_val_previous.items = dataset_val.items[:-1]

ds_split = \
    {"train": dataset_train_previous, 'train_next': dataset_train_next, "val": dataset_val_previous, 'val_next': dataset_val_next}




# evaluation metices
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
        mask = truths[i] < 255
        # print(mask)
        class_id = torch.max(output[i], dim=0).indices  # output[i].shape = (num_class, height, width)
        
        for j in range(class_num):

            pred_mask = class_id == j
            true_mask = truths[i] == j
            intersection[j] = (pred_mask & true_mask & mask).sum()
            union[j] = ((pred_mask | true_mask) & mask).sum()

        # in case we have zero on the denominator, use max to invoid a zero
        IOU.append(sum(intersection) / max(sum(union), 1e-5))
    iou = sum(IOU) / batch_num

    return iou


# evaluation metrice of depth estimation
def absRel(pred_depth: torch.Tensor, true_depth: torch.Tensor) -> float:
    pred_depth = pred_depth.detach().cpu()
    true_depth = true_depth.detach().cpu()
    return torch.mean(torch.abs(pred_depth - true_depth) / true_depth)


class Trainer:
    def __init__(
        self, model: nn.Module, ds_split: Dict[str, SemKittiDataset], lr, p_model
    ):
        # Choose a device to run training on. Ideally, you have a GPU available to accelerate the training process.
        self.p_model = p_model / "checkpoint"
        self.p_model.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move the model onto the target device
        self.model = model.to(self.device)

        # Store the dataset split separately
        self.ds_split = ds_split

        # use Adam to optimize and Cross Entropy Loss to train
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)    # 1.0
        # self.criterion = DeepLabCE(ignore_label=255)
        # self.criterion = DeepLabCE(ignore_label=255, top_k_percent_pixels=0.7) # 1.0, 0.8, 0.6
        # self.criterion_depth = nn.MSELoss()
        self.criterion_depth = DepthLoss()
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)

    def loss_instance(
        self,
        center_heatmap_pred,
        center_heatmap_gt,
        offset_pred,
        offset_gt,
        instance_mask,
        mask_center,
        mask_offset,
    ):
        # Compute MSE loss for center heatmap prediction
        center_loss = F.mse_loss(
            center_heatmap_pred, center_heatmap_gt, reduction="none"
        )
        # center_loss = (center_loss * instance_mask)[mask_center].mean()
        # center_loss = (center_loss)[mask_center].mean()
        center_loss = (center_loss).mean()

        # Compute L1 loss for offset prediction
        offset_loss = F.l1_loss(offset_pred, offset_gt, reduction="none")
        offset_loss = (offset_loss)[mask_offset].mean()

        return center_loss, offset_loss
    
    def loss_instance_next(self,offset_pred,offset_gt, mask_offset):
        offset_next_loss = F.l1_loss(offset_pred, offset_gt, reduction="none")
        offset_next_loss = (offset_next_loss)[mask_offset].mean()
        return offset_next_loss

    def train_epoch(self, dl: DataLoader, dl_next: DataLoader):
        # Put the model in training mode
        self.model.train()

        # Store each step's accuracy and loss for this epoch
        epoch_metrics = {
            "loss_seg": [],
            "loss_depth": [],
            "loss_center_offset": [],
            "loss_center_heatmap": [],
            "loss_center_offset_next": [],
            "loss": [],
        }

        # Create a progress bar using TQDM
        sys.stdout.flush()
        with tqdm(total=len(self.ds_split["train"]), desc=f"Training") as pbar:
            # Iterate over the training dataset
            for (sequence, inputs, truths), (sequence_next, inputs_next, truths_next) in zip(dl, dl_next):
                if sequence != sequence_next:
                    continue
                # save for multi-task
                truth_depth = truths[0] / 256.0
                truth_segment = truths[1].squeeze(dim=1)
                truth_instance = truths[2]
                truths_instance_next = truths_next[2]
                self.optimizer.zero_grad()

                # Move the inputs and truths to the target device
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                inputs_next = inputs_next.to(device=self.device, dtype=torch.float32)
                inputs.required_grad = True
                truth_segment = truth_segment.to(device=self.device, dtype=torch.long)
                truth_depth = truth_depth.to(device=self.device, dtype=torch.float32)

                # make mask due to sparsity
                mask_depth = truth_depth > 0
                mask_instance = truth_instance > 0
                mask_instance_next = truths_instance_next > 0
                mask_offset = torch.cat((mask_instance, mask_instance), dim=1)
                mask_offset_next = torch.cat((mask_instance_next, mask_instance_next), dim=1)

                # generate offset and center heatmap
                center, offset = _generate_gt_center_and_offset(batch_instance_labels=truth_instance)
                _, offset_next = _generate_gt_center_and_offset(batch_instance_labels=truths_instance_next)
                truth_instance = truth_instance.to(device=self.device, dtype=torch.float32)

                center = center.to(device=self.device, dtype=torch.float32)
                offset = offset.to(device=self.device, dtype=torch.float32)
                offset_next = offset_next.to(device=self.device, dtype=torch.float32)

                # Run model on the inputs
                (output_seg,output_depth,center_prediction,center_regression,center_regression_next) = self.model(inputs, inputs_next)

                # Perform backpropagation
                loss_seg = self.criterion(output_seg, truth_segment)
                loss_depth = self.criterion_depth(
                    output_depth[mask_depth], truth_depth[mask_depth]
                )
                center_loss, offset_loss = self.loss_instance(center_prediction,center,center_regression,offset,truth_instance,mask_instance,mask_offset)
                offset_next_loss = self.loss_instance_next(center_regression_next, offset_next, mask_offset_next)

                # weighted loss
                loss = (loss_depth + loss_seg + 200 * center_loss + 0.1 * offset_loss + 0.1 * offset_next_loss)
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                self.optimizer.step()
                self.scheduler.step()

                # Store the metrics of this step
                step_metrics = {
                    "loss_seg": loss_seg.item(),
                    "loss_depth": loss_depth.item(),
                    "loss_center_offset": offset_loss.item(),
                    "loss_center_heatmap": center_loss.item(),
                    "loss_center_offset_next": offset_next_loss.item(),
                    "loss": loss.item(),
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(list(inputs.shape)[0])

                # Add to epoch's metrics
                for k, v in step_metrics.items():
                    epoch_metrics[k].append(v)

        sys.stdout.flush()

        # Return metrics
        return epoch_metrics

    def val_epoch(self, dl: DataLoader, dl_next: DataLoader):
        # Put the model in evaluation mode
        self.model.eval()

        # Store the total loss and accuracy over the epoch
        amount = 0
        total_loss_seg = 0
        total_loss_depth = 0
        total_loss_center_heatmap = 0
        total_loss_center_offset = 0
        total_loss = 0
        total_accuracy = 0
        total_absRel = 0
        total_loss_center_offset_next = 0

        # Create a progress bar using TQDM
        sys.stdout.flush()
        with torch.no_grad(), tqdm(
            total=len(self.ds_split["val"]), desc=f"Validation"
        ) as pbar:
            # Iterate over the validation dataloader
            for (sequence, inputs, truths), (sequence_next, inputs_next, truths_next) in zip(dl, dl_next):
                if sequence != sequence_next:
                    continue
                # save for multi-task
                truth_depth = truths[0] / 256.0
                truth_segment = truths[1].squeeze(dim=1)
                truth_instance = truths[2]
                truths_instance_next = truths_next[2]

                # Move the inputs and truths to the target device
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                inputs_next = inputs_next.to(device=self.device, dtype=torch.float32)
                inputs.required_grad = True
                truth_segment = truth_segment.to(device=self.device, dtype=torch.long)
                truth_depth = truth_depth.to(device=self.device, dtype=torch.float32)

                # make mask due to sparsity
                mask_depth = truth_depth > 0
                mask_instance = truth_instance > 0
                mask_instance_next = truths_instance_next > 0
                mask_offset = torch.cat((mask_instance, mask_instance), dim=1)
                mask_offset_next = torch.cat((mask_instance_next, mask_instance_next), dim=1)

                # generate offset and center heatmap
                center, offset = _generate_gt_center_and_offset(batch_instance_labels=truth_instance)
                _, offset_next = _generate_gt_center_and_offset(batch_instance_labels=truths_instance_next)
                truth_instance = truth_instance.to(device=self.device, dtype=torch.float32)

                center = center.to(device=self.device, dtype=torch.float32)
                offset = offset.to(device=self.device, dtype=torch.float32)
                offset_next = offset_next.to(device=self.device, dtype=torch.float32)

                # Run model on the inputs
                (output_seg,output_depth,center_prediction,center_regression,center_regression_next) = self.model(inputs, inputs_next)

                # Perform backpropagation
                loss_seg = self.criterion(output_seg, truth_segment)
                loss_depth = self.criterion_depth(
                    output_depth[mask_depth], truth_depth[mask_depth]
                )
                center_loss, offset_loss = self.loss_instance(center_prediction,center,center_regression,offset,truth_instance,mask_instance,mask_offset)
                offset_next_loss = self.loss_instance_next(center_regression_next, offset_next, mask_offset_next)

                # weighted loss
                loss = (loss_depth + loss_seg + 200 * center_loss + 0.1 * offset_loss + 0.1 * offset_next_loss)
                if torch.isnan(loss):
                    continue

                # Store the metrics of this step
                step_metrics = {
                    "loss_seg": loss_seg.item(),
                    "loss_depth": loss_depth.item(),
                    "loss_center_offset": offset_loss.item(),
                    "loss_center_heatmap": center_loss.item(),
                    "loss_center_offset_next": offset_next_loss.item(),
                    "loss": loss.item(),
                    "accuracy_seg": compute_iou(output_seg, truth_segment),#output_seg_2D
                    "absRel": absRel(output_depth[mask_depth], truth_depth[mask_depth]),
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(list(inputs.shape)[0])

                amount += 1
                total_loss_center_heatmap += step_metrics["loss_center_heatmap"]
                total_loss_center_offset += step_metrics["loss_center_offset"]
                total_loss_depth += step_metrics["loss_depth"]
                total_loss_seg += step_metrics["loss_seg"]
                total_loss_center_offset_next += step_metrics["loss_center_offset_next"]
                total_loss += step_metrics["loss"]
                total_accuracy += step_metrics["accuracy_seg"]
                total_absRel += step_metrics["absRel"]
        sys.stdout.flush()

        # Print mean of metrics
        total_loss /= amount
        total_accuracy /= amount
        total_absRel /= amount
        total_loss_center_heatmap /= amount
        total_loss_center_offset /= amount
        total_loss_depth /= amount
        total_loss_seg /= amount
        total_loss_center_offset_next/= amount

        print(
            f"Validation loss is {total_loss/amount}, Validation loss_center_heatmap is {total_loss_center_heatmap/amount}, Validation loss_center_offset_next is {total_loss_center_offset_next/amount}, Validation loss_center_offset is {total_loss_center_offset/amount}, Validation loss_depth is {total_loss_depth/amount}, Validation loss_seg is {total_loss_seg/amount}, validation accuracy_seg is {total_accuracy}, validation absRel error is {total_absRel/amount}")

        # Return mean loss and accuracy
        return {
            "loss_seg": [total_loss_seg],
            "loss_depth": [total_loss_depth],
            "loss_center_offset": [total_loss_center_offset],
            "loss_center_heatmap": [total_loss_center_heatmap],
            "loss_center_offset_next": [total_loss_center_offset_next],
            "loss": [total_loss],
            "accuracy_seg": [total_accuracy],
            "absRel": [total_absRel],
        }

    # def fit(self, epochs: int, batch_size: int):
    #     # Initialize Dataloaders for the `train` and `val` splits of the dataset.
    #     # A Dataloader loads a batch of samples from the each dataset split and concatenates these samples into a batch.
    #     # dl_train = DataLoader(ds_split["train"], batch_size=batch_size, shuffle=True)
    #     # dl_val = DataLoader(ds_split["val"], batch_size=batch_size, drop_last=True)

    #     # use subset for test
    #     # Determine the number of samples to use for training
    #     num_samples = len(ds_split["train"])
    #     num_val = len(ds_split["val"])
    #     num_train_samples = num_samples // 100
    #     num_val_samples = num_val // 100

    #     # Create a subset of the training dataset with 1/100 samples
    #     train_subset = torch.utils.data.Subset(
    #         ds_split["train"], list(range(num_train_samples))
    #     )
    #     train_subset_next = torch.utils.data.Subset(
    #         ds_split["train_next"], list(range(num_train_samples))
    #     )
    #     val_subset = torch.utils.data.Subset(
    #         ds_split["val"], list(range(num_val_samples))
    #     )
    #     val_subset_next = torch.utils.data.Subset(
    #         ds_split["val_next"], list(range(num_val_samples))
    #     )

    #     seed = 42
    #     torch.manual_seed(seed)

    #     # Create the DataLoader for the training subset
    #     dl_train_subset = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),drop_last=True)
    #     dl_train_subset_next = DataLoader(train_subset_next, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),drop_last=True)
    #     dl_val_subset = DataLoader(val_subset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),drop_last=True)
    #     dl_val_subset_next = DataLoader(val_subset_next, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),drop_last=True)

    #     # Store metrics of the training process (plot this to gain insight)
    #     df_train = pd.DataFrame()
    #     df_val = pd.DataFrame()

    #     # Train the model for the provided amount of epochs
    #     for epoch in range(1, epochs + 1):
    #         print(f"Epoch {epoch}")

    #         metrics_train = self.train_epoch(dl_train_subset, dl_train_subset_next)
    #         df_train = pd.concat(
    #             [
    #                 df_train,
    #                 pd.DataFrame(
    #                     {
    #                         "epoch": [
    #                             epoch for _ in range(len(metrics_train["loss_seg"]))
    #                         ],
    #                         **metrics_train,
    #                     }
    #                 ),
    #             ],
    #             ignore_index=True,
    #         )

    #         if epoch % 5 == 0:
    #             model_name = f"checkpoint_epoch_{epoch}.pth"
    #             torch.save(self.model.state_dict(), self.p_model / model_name)

    #         metrics_val = self.val_epoch(dl_val_subset, dl_val_subset_next)
    #         df_val = pd.concat(
    #             [
    #                 df_val,
    #                 pd.DataFrame(
    #                     {
    #                         "epoch": [
    #                             epoch for _ in range(len(metrics_val["loss_seg"]))
    #                         ],
    #                         **metrics_val,
    #                     }
    #                 ),
    #             ],
    #             ignore_index=True,
    #         )

    #     # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.
    #     return df_train, df_val

    def fit(self, epochs: int, batch_size:int):
        # Initialize Dataloaders for the `train` and `val` splits of the dataset.
        # A Dataloader loads a batch of samples from the each dataset split and concatenates these samples into a batch.
        # dl_train = DataLoader(ds_split["train"], batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(ds_split["val"], batch_size=batch_size, drop_last=True)
        dl_val_next = DataLoader(ds_split["val_next"], batch_size=batch_size, drop_last=True)

        # seed = 42
        # torch.manual_seed(seed)

        # dl_train_next = DataLoader(ds_split["train"], batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True)
        # dl_train = DataLoader(ds_split["train_next"], batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True)


        # Store metrics of the training process (plot this to gain insight)
        df_train = pd.DataFrame()
        df_val = pd.DataFrame()

        # Train the model for the provided amount of epochs
        for epoch in range(1, epochs+1):

            seed = 42 + epoch - 1
            torch.manual_seed(seed)
            dl_train = DataLoader(ds_split["train"], batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True)
            dl_train_next = DataLoader(ds_split["train_next"], batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), drop_last=True)

            print(f'Epoch {epoch}')

            metrics_train = self.train_epoch(dl_train, dl_train_next)
            df_train = pd.concat([df_train,pd.DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss_seg"]))], **metrics_train})], ignore_index=True)

            if epoch % 5 == 0:
                model_name = f"checkpoint_epoch_{epoch}.pth"
                torch.save(self.model.state_dict(), self.p_model / model_name)

            metrics_val = self.val_epoch(dl_val, dl_val_next)
            df_val = pd.concat([df_val,pd.DataFrame({'epoch': [epoch for _ in range(len(metrics_val["loss_seg"]))], **metrics_val})], ignore_index=True)

        # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.
        return df_train, df_val


# create folder
folder_name = "DVPS_two_frame_0622_CE"
p_model = Path() / "Train" / folder_name
p_model.mkdir(parents=True, exist_ok=True)

# free memory
torch.cuda.empty_cache()

# Train the passthrough network
print("Training...")
model = Model()

trainer = Trainer(model, ds_split, lr=1e-3, p_model=p_model)
train_results, val_results = trainer.fit(epochs=60, batch_size=8)


# Pick sample 819 and 820
idx = 819
sequence, input, truth = ds_split["val"][idx]
sequence_next, input_next, truth_next = ds_split["val"][idx+1]

# Push through our network
model.eval()
model = model.cpu()
output = model(input.unsqueeze(0), input_next.unsqueeze(0))
# save model in the folder
model_name = f"model.pth"
torch.save(model.state_dict(), (p_model / model_name))

# save dataframe file
train_results.to_csv((p_model / f"train_results.csv"))
val_results.to_csv((p_model / f"val_results.csv"))

### display in HTML format

# Display the input, output and truth tensors of semantic segmantation
template_table = "<table><thead><tr><th>Tensor</th><th>Shape</th><th>Image</th></tr></thead><tbody>{0}</tbody></table>"
template_row = (
    '<tr><td>{0}</td><td>{1}</td><td><img src="data:image/png;base64,{2}"/></td></tr>'
)

input_img = TF.to_pil_image(input)
truth_img = ds_split["val"].to_image(truth)
img_semantic = ds_split["val"].masks_to_indices(output[0]).to(dtype=torch.int32)

output_img = ds_split["val"].to_image([(img_semantic).clone() for _ in range(3)])
rows = []
for name, tensor, img in [
    ("Input", input, input_img),
    ("Output_semantic", output[0], output_img[1]),
    ("Target_semantic", truth[1], truth_img[1]),
]:
    with BytesIO() as b:
        img.save(b, format="png")
        rows.append(
            template_row.format(
                name, list(tensor.shape), b64encode(b.getvalue()).decode("utf-8")
            )
        )

# Render HTML table
table = template_table.format("".join(rows))

# save table
html_content = table
html_filename = p_model / "output_semantic.html"
with open(html_filename, "w") as f:
    f.write(html_content)

filename_sem = p_model / "output_semantic.png"
output_img[1].save(filename_sem)

filename_sem_truth = p_model / "truth_semantic.png"
truth_img[1].save(filename_sem_truth)


# depth estimation and its truth
out_depth = (output[1][0] * 256.0).to(dtype=torch.int)
output_img_depth = TF.to_pil_image(out_depth, "I")
filename_depth = p_model / "output_depth.png"
output_img_depth.save(filename_depth)

out_depth_no = (output[1][0]).to(dtype=torch.int)
output_img_depth_no = TF.to_pil_image(out_depth_no, "I")
filename_depth_no = p_model / "output_depth.png"
output_img_depth_no.save(filename_depth_no)


filename_depth_truth = p_model / "truth_depth.png"
truth_img[0].save(filename_depth_truth)


###############################
# display heatmap and its truth

center_image = output[2].squeeze().detach().numpy()
plt.figure()
plt.imshow(center_image, cmap="hot", interpolation="nearest")
plt.colorbar()
filename_heatmap = p_model / "output_heatmap.png"
plt.savefig(filename_heatmap)

center_truth, offset_truth = _generate_gt_center_and_offset(
    batch_instance_labels=truth[2].unsqueeze(0)
)

center_truth_image = center_truth.squeeze().detach().numpy()
plt.figure()
plt.imshow(center_truth_image, cmap="hot", interpolation="nearest")
plt.colorbar()
filename_heatmap_truth = p_model / "truth_heatmap.png"
plt.savefig(filename_heatmap_truth)


### plot and save figures ###
# plot and save loss of semantic segmentation
train_loss_seg = []
val_loss_seg = []
painted = "loss_seg"
for i in range(1, train_results["epoch"].max()):
    train_loss_seg.append(train_results[train_results["epoch"] == i][painted].mean(0))
    val_loss_seg.append(val_results[val_results["epoch"] == i][painted].mean(0))

plt.figure()
plt.plot(train_loss_seg, "b-")
plt.plot(val_loss_seg, "r--")
plt.title(painted)
plt.ylabel(painted)
plt.xlabel("epoch")
plt.legend(["train", "val"])
pic_name = f"loss_seg.png"
plt.savefig((p_model / pic_name))

# plot and save IoU accuracy of semantic segmentation
# train_IoU = []
val_IoU = []
painted = "accuracy_seg"
for i in range(1, train_results["epoch"].max()):
    # train_IoU.append(train_results[train_results['epoch']==i][painted].mean(0))
    val_IoU.append(val_results[val_results["epoch"] == i][painted].mean(0))

plt.figure()
# plt.plot(train_IoU,'b-')
plt.plot(val_IoU, "r--")
plt.title(painted)
plt.ylabel(painted)
plt.xlabel("epoch")
plt.legend(["val"])
pic_name1 = f"accuracy_seg.png"
plt.savefig((p_model / pic_name1))

# plot and save loss of depth estimation
train_loss_depth = []
val_loss_depth = []
painted = "loss_depth"
for i in range(1, train_results["epoch"].max()):
    train_loss_depth.append(train_results[train_results["epoch"] == i][painted].mean(0))
    val_loss_depth.append(val_results[val_results["epoch"] == i][painted].mean(0))

plt.figure()
plt.plot(train_loss_depth, "b-")
plt.plot(val_loss_depth, "r--")
plt.title(painted)
plt.ylabel(painted)
plt.xlabel("epoch")
plt.legend(["train", "val"])
pic_name2 = f"loss_depth.png"
plt.savefig((p_model / pic_name2))

# plot and save loss of center offset
train_loss_offset = []
val_loss_offset = []
painted = "loss_center_offset"
for i in range(1, train_results["epoch"].max()):
    train_loss_offset.append(
        train_results[train_results["epoch"] == i][painted].mean(0)
    )
    val_loss_offset.append(val_results[val_results["epoch"] == i][painted].mean(0))

plt.figure()
plt.plot(train_loss_offset, "b-")
plt.plot(val_loss_offset, "r--")
plt.title(painted)
plt.ylabel(painted)
plt.xlabel("epoch")
plt.legend(["train", "val"])
pic_name3 = f"loss_center_offset.png"
plt.savefig((p_model / pic_name3))


# plot and save loss of center offset
train_loss_offset_next = []
val_loss_offset_next = []
painted = "loss_center_offset_next"
for i in range(1, train_results["epoch"].max()):
    train_loss_offset_next.append(train_results[train_results["epoch"] == i][painted].mean(0))
    val_loss_offset_next.append(val_results[val_results["epoch"] == i][painted].mean(0))

plt.figure()
plt.plot(train_loss_offset_next, "b-")
plt.plot(val_loss_offset_next, "r--")
plt.title(painted)
plt.ylabel(painted)
plt.xlabel("epoch")
plt.legend(["train", "val"])
pic_name5 = f"loss_center_offset_next.png"
plt.savefig((p_model / pic_name5))


# plot and save loss of center prediction
train_loss_center = []
val_loss_center = []
painted1 = "loss_center_heatmap"
for i in range(1, train_results["epoch"].max()):
    train_loss_center.append(
        train_results[train_results["epoch"] == i][painted1].mean(0)
    )
    val_loss_center.append(val_results[val_results["epoch"] == i][painted1].mean(0))

plt.figure()
plt.plot(train_loss_center, "b-")
plt.plot(val_loss_center, "r--")
plt.title(painted1)
plt.ylabel(painted1)
plt.xlabel("epoch")
plt.legend(["train", "val"])
pic_name4 = f"loss_center_heatmap.png"
plt.savefig((p_model / pic_name4))

# plot and save absRel
train_loss_center = []
val_loss_center = []
painted1 = "absRel"
for i in range(1, train_results["epoch"].max()):
    # train_loss_center.append(train_results[train_results['epoch']==i][painted1].mean(0))
    val_loss_center.append(val_results[val_results["epoch"] == i][painted1].mean(0))

plt.figure()
plt.plot(train_loss_center, "b-")
plt.plot(val_loss_center, "r--")
plt.title(painted1)
plt.ylabel(painted1)
plt.xlabel("epoch")
plt.legend(["val"])
pic_name4 = f"absRel.png"
plt.savefig((p_model / pic_name4))
