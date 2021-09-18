import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.utils import resize_image, crop_square

class CatAndDogDataset(Dataset):
    ''' load CatAndDogDataset for training with annotation
    Args: 
        csv_file: (string) annotation file for dataset
        transform: (object) transformation of data for training
    
    '''

    def __init__(self, csv_file, transform=None, dir_train_images=None):
        self.annotations = pd.read_csv(csv_file, index_col = 0)
        self.transform = transform
        self.base_dir = dir_train_images
        self.images = self.annotations.iloc[:, 0].values.tolist()
        self.labels = self.annotations.iloc[:, 1].values.tolist()

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        image = self.images[index]
        image_path = os.path.join(self.base_dir, image)
        label = torch.tensor(int(self.labels[index]), dtype = torch.float32)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label 


class DatasetTransform(object):
    def __init__(self, resize=None, crop_square=True):
        self.resize = resize
        self.crop = crop_square
    
    def __call__(self, image, label):
        
        if self.crop:
            image = crop_square(image)
        if self.resize:
            image = resize_image(image, self.resize)
        
        normalize = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize([0.484, 0.456, 0.406],[0.229, 0.224, 0.225])])
        
        image_after = normalize(image)
        return image_after, label

class CatAndDogDatasetTest(Dataset):
    '''Load CatDogDataset without annotations
    '''
    def __init__(self, base_dir, transform):
        self.dir = base_dir
        self.transform = transform
    def __len__(self):
        return (len(os.listdir(self.dir)))
    
    def __getitem__(self, index): 
        files = os.listdir(self.dir)
        file = files[index]
        image_path = os.path.join(self.dir, file)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image, _ = self.transform(image, None)
        
        return image, file

if __name__ == "__main__":
    csv_file = os.path.join('small_dataset', 'small_dataset.csv')
   
    dir_train = os.path.join('small_dataset', 'images')
    transformer = DatasetTransform(resize=(224,224))
    dataset = CatAndDogDataset(csv_file, transformer, dir_train)
    
    #print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1])
    





    
