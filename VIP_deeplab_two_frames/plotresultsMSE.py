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


# evaluation metrice of depth estimation
def absRel(pred_depth: torch.Tensor, true_depth: torch.Tensor) -> float:
    pred_depth = pred_depth.detach().cpu()
    true_depth = true_depth.detach().cpu()
    return torch.mean(torch.abs(pred_depth - true_depth) / true_depth)


# create folder
folder_name = "results_MSE"
p_model = Path() / "Train" / folder_name
p_model.mkdir(parents=True, exist_ok=True)

current_dir = os.getcwd()  
target_dir = os.path.join(current_dir, "Train", "DVPS_two_frame_0622_MSE", "checkpoint")    # change this for new folder
model_path = os.path.join(target_dir, "checkpoint_epoch_50.pth")

model = Model()
model.load_state_dict(torch.load(model_path))
model.eval()

# Pick sample 819 and 820
idx = 819
sequence, input, truth = ds_split["val"][idx]
sequence_next, input_next, truth_next = ds_split["val"][idx+1]

# Push through our network
model.eval()
model = model.cpu()
output = model(input.unsqueeze(0), input_next.unsqueeze(0))
# # save model in the folder
# model_name = f"model.pth"
# torch.save(model.state_dict(), (p_model / model_name))

# # save dataframe file
# train_results.to_csv((p_model / f"train_results.csv"))
# val_results.to_csv((p_model / f"val_results.csv"))

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