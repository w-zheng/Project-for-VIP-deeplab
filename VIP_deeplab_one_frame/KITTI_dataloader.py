import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Optional, Tuple, List
import torchvision.transforms.functional as TF
import random
from torchvision.transforms import transforms
import re
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from dataclasses import dataclass


@dataclass
class SemKITTI_Sample:
    sequence: str
    frame_id: str
    focal_length_real: str

    @property
    def id(self):
        return self.sequence + "_" + self.frame_id
    
    @property
    def focal_length(self):
        return self.focal_length_real

    @staticmethod
    def from_filename(filename: str):
        match = re.match(r"^(\d+)_(\d+)_depth_(\d+\.\d+).png$", filename, re.I)
        return SemKITTI_Sample(match.group(1), match.group(2),match.group(3))




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


import os
from PIL import Image
from typing import List, Tuple
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset


@dataclass
class SemKITTI_Sample:
    sequence: str
    frame_id: str
    focal_length_real: str

    @property
    def id(self):
        return self.sequence + "_" + self.frame_id
    
    @property
    def focal_length(self):
        return self.focal_length_real

    @staticmethod
    def from_filename(filename: str):
        match = re.match(r"^(\d+)_(\d+)_depth_(\d+\.\d+).png$", filename, re.I)
        return SemKITTI_Sample(match.group(1), match.group(2),match.group(3))

class SemKittiDataset(Dataset):

    def __init__(self, dir_input: str, classes: Dict):
        super().__init__()
        self.dir_input = dir_input
        self.classes = classes
        self.items = []
        for filename in sorted(os.listdir(self.dir_input)):
            if "depth" in filename:
                self.items.append(SemKITTI_Sample.from_filename(filename))
        assert len(self.items) > 0, f"No items found in {self.dir_input}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> (torch.Tensor, List[torch.Tensor]):
        sample = self.items[i]
        input = self.load_file(sample,"input")
        truth_depth = self.load_file(sample,"depth")
        truth_segment = self.load_file(sample,"class_segment")
        truth_instance = self.load_file(sample,"insatnce_segment")
        return self.transform(input, [truth_depth,truth_segment,truth_instance])

    def load_file(self, sample: SemKITTI_Sample,image_type: str) -> Image:
        file_name = ""
        if image_type == "input":
            file_name = f'{sample.id}_leftImg8bit.png'
        elif image_type == "depth":
            file_name = f'{sample.id}_depth_{sample.focal_length}.png'
        elif image_type == "class_segment":
            file_name = f'{sample.id}_gtFine_class.png'
        elif image_type == "insatnce_segment":
            file_name = f'{sample.id}_gtFine_instance.png'
        else:
            raise Exception("image_type incorrect")
        path = os.path.join(self.dir_input, file_name)
        return Image.open(path)

    def transform(self, img: Image.Image, mask: List[Image.Image]) -> (torch.Tensor, List[torch.Tensor]):
    # def transform(self, img: Image.Image, mask: Optional[Image.Image]) -> (torch.Tensor, torch.Tensor):
        width, height = (1280, 352)#(1216, 352)(1240, 368)
        img = img.resize((width, height), Image.NEAREST)
        for i in range(len(mask)):
            mask[i] = mask[i].resize((width, height), Image.NEAREST)
        img = TF.to_tensor(img)
        truth = []
        for truth_image in mask:
            truth.append(TF.to_tensor(truth_image))  
        return img, truth
    
    def masks_to_indices(self, masks: torch.Tensor) -> torch.Tensor:
        _, indices = masks.softmax(dim=1).max(dim=1)
        return indices

    def to_image(self, indices: List[torch.Tensor]) -> List[Image.Image]:
        img_depth = TF.to_pil_image(indices[0].cpu(), 'I')
        img_instance = np.zeros([indices[2].shape[1],indices[2].shape[2],3],dtype=np.uint8)
        colors = {0: np.array([0,0,0])}
        for i in range(indices[2].shape[1]):
            for j in range(indices[2].shape[2]):
                instance = indices[2][0][i][j].item()
                #print(instance)
                if instance not in colors:
                    colors[instance] = np.array([random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)])
                img_instance[i][j] = colors[instance]
        img_instance = Image.fromarray(img_instance)

        img_segment = np.zeros([indices[1].shape[1],indices[1].shape[2],3],dtype=np.uint8)
        for i in range(indices[1].shape[1]):
            for j in range(indices[1].shape[2]):
                class_id = indices[1][0][i][j].item()
                img_segment[i][j] = classes[class_id].color
        img_segment = Image.fromarray(img_segment)

        return [img_depth,img_segment,img_instance]
    
    def get_instance_labels(self, index: int) -> torch.Tensor:
        _, truth = self.__getitem__(index)                               
        return truth[2]  



    