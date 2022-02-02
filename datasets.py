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
        self.map = category_map
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.X[index], self.y[index]

    
    def visualize_random(self,no_samples, size=80):
        
        import matplotlib.pyplot as plt
        import random

        figsize = size/no_samples
        
        fig, axs = plt.subplots(1,no_samples,figsize=(figsize,figsize))

        for i in range(no_samples):
            if no_samples > 1:
                axs[i].axis('off')
            else:
                axs.axis('off')
            index = random.randint(0,len(self))
            img_tens, label = self[index]
            img_arr = img_tens.cpu().detach().numpy()[0,0,:,:]
            label_index = int(label.argmax())
            label_name = list(self.map.keys())[label_index]
            axs[i].imshow(img_arr, cmap = 'gray')
            axs[i].title.set_text(label_name)
        plt.show()


    def visualize(self,index):
        
        import matplotlib.pyplot as plt
        import random
        
        fig, axs = plt.subplots(1,figsize=(20,20))

        img_tens, label = self[index]
        img_arr = img_tens.cpu().detach().numpy()[0,0,:,:]
        label_index = int(label.argmax())
        label_name = list(self.map.keys())[label_index]
        axs.imshow(img_arr, cmap = 'gray')
        axs.title.set_text(label_name)
        plt.show()


    def plot_classes(self):

        import matplotlib.pyplot as plt
        import numpy as np

        counts = {'buildings':0, 'forest':0, 'glacier':0, 'mountain':0, 'sea':0, 'street':0}
        for entry in self.y:
            i = int(entry.argmax())
            label = list(counts.keys())[i]
            counts[label] +=1
        y_range = np.arange(len(counts))
        plt.barh(y_range, counts.values())
        plt.yticks(y_range,labels=list(counts.keys()))
        plt.xlabel('Number of Class Instances')
        plt.title('Class Distribution of Dataset')
        plt.show()



