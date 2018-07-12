import csv
import os
from os.path import expanduser
import pandas as pd
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

# we trust this dataset (I think?)
Image.MAX_IMAGE_PIXELS = None

data_dir = expanduser("~") +"/Data/artist/"
all_artist_data = data_dir + "all_artist_data.csv"
filtered = data_dir + "filtered.csv"
artist_train = data_dir + "train"
log_dir = data_dir + "logs/"
model_dir = data_dir + "models/"

label_to_artist = {}
artist_to_label = {}

def init_label_dicts():
    label_counter = 0

    with open(filtered, "r", encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, quotechar='\"')
        header = next(reader)
        for row in reader:
            if row[0] not in artist_to_label:
                label_to_artist[label_counter] = row[0]
                artist_to_label[row[0]] = label_counter
                label_counter += 1

def get_num_artists():
    return len(artist_to_label)

# run when imported
init_label_dicts()                

train_loader_transform = transforms.Compose([transforms.RandomCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
'''
We want to load in an transform our data into proper format. This involves implementing the 
Dataset asbtract class as well as instantiate dataloader classes with versions specific to our data and
our desired transformations. To follow along with the paper, we are going to randomly crop 224x244
images out of the training images. 
'''
class ArtistImageDataset(torch.utils.data.Dataset):
    def __init__(self,text_file,img_dir,transform=train_loader_transform):
        self.name_frame = pd.read_csv(text_file,sep=",",usecols=range(11,12))
        self.label_frame = pd.read_csv(text_file,sep=",",usecols=range(1))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.name_frame.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        labels = artist_to_label[self.label_frame.iloc[index, 0]]
        sample = {'images': image, 'labels': labels}

        return sample

def get_dataloaders(batch_size=64):
    dataset = ArtistImageDataset(text_file=filtered, img_dir = artist_train)
    #split into train, test, and validation sets
    num_imgs = len(dataset)
    indices = list(range(num_imgs))
    test_indices = np.random.choice(indices, size=num_imgs//10, replace=False)
    train_indices = list(set(indices) - set(test_indices))
    indices = train_indices
    validation_indices = np.random.choice(indices, size=num_imgs//10, replace=False)
    train_indices = list(set(indices) - set(validation_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=test_sampler)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=validation_sampler)
    return train_loader, test_loader, val_loader

# image-showing code taken from the PyTorch tutorial
def show_sample_images():

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(train_loader)
    sample = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(sample['images']))
    # print labels
    print([label_to_artist[s.data.numpy()[()]] for s in sample['labels']])