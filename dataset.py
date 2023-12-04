from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

class JAADDataset(Dataset):
    def __init__(self, root_dir,image_dir,ped_dir,label_dir):
        self.root_dir = root_dir
        self.image_dir=image_dir
        self.ped_dir=ped_dir
        self.label_dir=label_dir
        self.path2ima=os.path.join(self.root_dir,self.image_dir)
        self.path2ped=os.path.join(self.root_dir,self.ped_dir)
        self.path2label=os.path.join(self.root_dir,self.label_dir)
        self.image_list = os.listdir(self.path2ima)
        self.ped_list=os.listdir(self.path2ped)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        sample_im=os.path.join(self.path2ima,self.image_list[idx])
        sample_pd=os.path.join(self.path2ped,self.ped_list[idx])
        sample_label=self.readintxt(self.path2label)[idx]


        #readin the scene image
        image_1 = Image.open(sample_im)
        image_1 = self.transform(image_1)

        #readin the pedestrian image
        image_ped=Image.open(sample_pd)
        image_ped=self.transform(image_ped)

        return image_1,image_ped, sample_label
    #read in the txt file，如果使用list，则替换这部分，使用list读入返回list就行
    def readintxt(self,label_dir):
        label = []
        with open(label_dir) as f:
            s = f.readline()
            while True:
                a = s.replace('\n', '')
                label.append(int(a))
                s = f.readline()
                if s == '' or s == '\n':
                    break
        return label