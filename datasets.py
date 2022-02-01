from torch.utils.data import Dataset
import torch
from .colorization.colorizers import load_img, preprocess_img
import os


def normalize_l(in_l):

    l_cent = 50.
    l_norm = 100.
    ab_norm = 110.
    return (in_l-l_cent)/l_norm



class intelDataset(Dataset):
    
    
    def __init__(self, data_path):

        'Initialization'
        class_paths = [data_path + '/' + path for path in os.listdir(data_path) if path[0] != '.']
        category_map = {'buildings':0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
        train_x = []
        train_y = []
        for path in class_paths:
            class_name = path.split('/')[-1]
            image_paths = [path + '/' + st for st in os.listdir(path)]
            for image in image_paths:
                class_index = category_map[class_name]
                y_value = torch.zeros(6)
                y_value[class_index] = 1
                train_y.append(y_value)
                img = load_img(image)
                (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
                normal_img = normalize_l(tens_l_rs)
                train_x.append(normal_img)
        self.X = train_x
        self.y = train_y
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.X[index], self.y[index]