import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Optional, Tuple, List
import torchvision.transforms.functional as TF
import random
from torchvision.transforms import transforms
import re
from io import BytesIO
from base64 import b64encode
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os


# def generate_instances_tensor(ds_split_train, sample):
#     # print(sample.shape)

#     images = [img.convert('RGB') for img in ds_split_train.to_image(sample)]
#     instances_img = images[2]
#     instances_array = np.array(instances_img)
#     instances_tensor = torch.from_numpy(instances_array)

#     return instances_tensor


# def _generate_gt_center_and_offset(instance_labels):
#     height = instance_labels.size(0)
#     width = instance_labels.size(1)

#     sigma = 8
#     center_pad_begin = int(round(3 * sigma + 1))
#     center_pad_end = int(round(3 * sigma + 2))
#     center_pad = center_pad_begin + center_pad_end

#     center = torch.zeros((height + center_pad, width + center_pad))
#     offset = torch.zeros((height, width, 2), dtype=torch.int32)

#     unique_ids = torch.unique(instance_labels)

#     for instance_id in unique_ids:
#         # Background pixels are ignored
#         if instance_id == 0:
#             continue

#         mask_indices = torch.nonzero(instance_labels == instance_id, as_tuple=False)
#         mask_y_index = mask_indices[:, 0]
#         mask_x_index = mask_indices[:, 1]

#         instance_area = mask_x_index.size(0)
#         if instance_area < 1:
#             continue

#         center_y = torch.mean(mask_y_index.float())
#         center_x = torch.mean(mask_x_index.float())

#         center_x = torch.round(center_x).to(torch.int32)
#         center_y = torch.round(center_y).to(torch.int32)

#         upper_left = (max(center_x - sigma, 0), max(center_y - sigma, 0))
#         bottom_right = (min(center_x + sigma + 1, width), min(center_y + sigma + 1, height))

#         indices_x, indices_y = torch.meshgrid(
#             torch.arange(upper_left[0], bottom_right[0]),
#             torch.arange(upper_left[1], bottom_right[1]))
#         indices = torch.stack([indices_y.reshape(-1),
#                                indices_x.reshape(-1)], dim=1)

#         gaussian = torch.exp(-0.5 * (((indices[:, 0] - center_y) / sigma) ** 2 + ((indices[:, 1] - center_x) / sigma) ** 2))
#         gaussian = gaussian.reshape(bottom_right[1] - upper_left[1], bottom_right[0] - upper_left[0])

#         center[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] += gaussian  
#         offset_y = center_y - mask_y_index.to(torch.int32)
#         offset_x = center_x - mask_x_index.to(torch.int32)
#         offset[mask_y_index, mask_x_index] = torch.stack([offset_y, offset_x], dim=1)

#     center = center[center_pad_begin:(center_pad_begin + height),
#                     center_pad_begin:(center_pad_begin + width)]
#     center = center.unsqueeze(-1)

#     return center, offset


def _generate_gt_center_and_offset(batch_instance_labels):
    batch_size = batch_instance_labels.size(0)
    height = batch_instance_labels.size(2)
    width = batch_instance_labels.size(3)
    mask_y_index = None
    mask_x_index = None
    sigma = 8
    center_pad_begin = int(round(3 * sigma + 1))
    center_pad_end = int(round(3 * sigma + 2))
    center_pad = center_pad_begin + center_pad_end

    center = torch.zeros((batch_size, height + center_pad, width + center_pad, 1), dtype=torch.float32)
    offset = torch.zeros((batch_size, height, width, 2), dtype=torch.int32)

    for b in range(batch_size):
        instance_labels = batch_instance_labels[b]

        unique_ids = torch.unique(instance_labels)

        for instance_id in unique_ids:
            # Background pixels are ignored
            if instance_id == 0:
                continue

            mask_indices = torch.nonzero(instance_labels == instance_id, as_tuple=False)
            mask_y_index = mask_indices[:, 1]
            mask_x_index = mask_indices[:, 2]

            instance_area = mask_x_index.size(0)
            if instance_area < 1:
                continue

            center_y = torch.mean(mask_y_index.float())
            center_x = torch.mean(mask_x_index.float())

            center_x = torch.round(center_x).to(torch.int32)
            center_y = torch.round(center_y).to(torch.int32)

            upper_left = (max(center_x - sigma, 0), max(center_y - sigma, 0))
            bottom_right = (min(center_x + sigma + 1, width + center_pad), min(center_y + sigma + 1, height + center_pad))

            indices_x, indices_y = torch.meshgrid(
                torch.arange(upper_left[0], bottom_right[0]),
                torch.arange(upper_left[1], bottom_right[1]),indexing="ij")
            indices = torch.stack([indices_y.reshape(-1),
                                   indices_x.reshape(-1)], dim=1)

            gaussian = torch.exp(-0.5 * (((indices[:, 0] - center_y) / sigma) ** 2 + ((indices[:, 1] - center_x) / sigma) ** 2))
            gaussian = gaussian.reshape(bottom_right[1] - upper_left[1], bottom_right[0] - upper_left[0])

            center[b, upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0], 0] += gaussian
            offset_y = center_y - upper_left[1] + mask_y_index.to(torch.int32)
            offset_x = center_x - upper_left[0] + mask_x_index.to(torch.int32)
            offset[b, mask_y_index, mask_x_index] = torch.stack([offset_y, offset_x], dim=1)

    center = center[:, center_pad_begin:center_pad_begin + height, center_pad_begin:center_pad_begin + width, :]
    center = center.permute(0, 3, 1, 2)        # 0-batch_size  1-1  2-height  3-width
    offset = offset.permute(0, 3, 1, 2)

    return center, offset
