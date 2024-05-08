from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd
import os
from PIL import Image
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self,data: pd.DataFrame,mode : str):
        self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(tuple(train_mean), tuple(train_std)) ])
        self.data = data
        self.mode = mode
        
    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        
        current_working_directory = os.getcwd()
        
        image = Image.open(str(current_working_directory) + '/' + str(self.data.iloc[index,0]))
        rgbimage = gray2rgb(image)
        transformed_image = self._transform(rgbimage)
        # access the labels, which are stored in the second and third column
        label = torch.tensor([float(self.data.iloc[index,1]),float(self.data.iloc[index,2])])
        return transformed_image,label        
        