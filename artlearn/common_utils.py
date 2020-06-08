import csv
import os
from os.path import expanduser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# we trust this dataset (I think?)
Image.MAX_IMAGE_PIXELS = None

DATA_DIR = os.path.join(expanduser('~'), 'data', 'artist')
ALL_ARTIST_DATA = os.path.join(DATA_DIR, 'all_artist_data.csv')
FILTERED = os.path.join(DATA_DIR, 'filtered.csv')
ARTIST_TRAIN = os.path.join(DATA_DIR, 'train')
LOG_DIR = os.path.join(DATA_DIR, 'logs')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

label_to_artist = {}
artist_to_label = {}


def fetch_image(image_name, as_arr=False):
    img = Image.open(os.path.join(ARTIST_TRAIN, image_name)).convert('RGB')
    if as_arr:
        return np.array(img)
    return img


def init_label_dicts():
    label_counter = 0

    with open(FILTERED, "r", encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, quotechar='\"')
        next(reader)
        for row in reader:
            if row[0] not in artist_to_label:
                label_to_artist[label_counter] = row[0]
                artist_to_label[row[0]] = label_counter
                label_counter += 1
    return label_to_artist, artist_to_label


def get_num_artists():
    return len(artist_to_label)


# run when imported
init_label_dicts()

train_loader_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


'''
We want to load in an transform our data into proper format. This involves
implementing the Dataset asbtract class as well as instantiating dataloader
classes with versions specific to our data and our desired transformations.
'''
class ArtistImageDataset(torch.utils.data.Dataset):
    def __init__(self, text_file, img_dir, transform=train_loader_transform):
        self.name_frame = pd.read_csv(text_file, sep=",",
                                      usecols=range(11, 12))
        self.label_frame = pd.read_csv(text_file, sep=",", usecols=range(1))
        self.img_dir = img_dir
        self.transform = transform
        self.label_to_artist, self.artist_to_label = init_label_dicts()

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, index):
        img_name = self.name_frame.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        labels = self.artist_to_label[self.label_frame.iloc[index, 0]]
        sample = {'images': image, 'labels': labels, 'names': img_name}

        return sample


def get_dataloaders(batch_size=64):
    """Create dataloaders for our dataset for train, test, and validation

    Parameters
    ----------
    batch_size: int

    Returns
    -------
    tuple of torch.utils.data.DataLoader
    """
    dataset = ArtistImageDataset(text_file=FILTERED, img_dir=ARTIST_TRAIN)
    # split into train, test, and validation sets
    num_imgs = len(dataset)
    indices = list(range(num_imgs))
    test_indices = np.random.choice(indices, size=num_imgs//10, replace=False)
    train_indices = list(set(indices) - set(test_indices))
    indices = train_indices
    validation_indices = np.random.choice(indices, size=num_imgs//10,
                                          replace=False)
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
def show_sample_images(train_loader):

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
    for artist, img in zip(sample['labels'], sample['names']):
        print(f'{label_to_artist[artist.data.numpy()[()]]}: {img}')


class ArtistLearner(object):
    '''Class to enable training of a torch.nn.Module

    Parameters
    ----------
    network: torch.nn.Module
    mode: str
    epochs: int
    train: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    lr: float
    momentum: float
    log_after:int
        how many iterations to log after. If falsy, do no logging
    log_path: str
    model_path: str
    model_name: str
        if left None, a name will be generated based on the model params
    '''
    def __init__(self, network, mode, epochs, train, test, val, lr=1e-3,
                 momentum=0.9, log_after=80, log_path=LOG_DIR,
                 model_path=MODEL_DIR, model_name=None):
        self.network = network
        self.mode = mode
        self.epochs = epochs
        self.train = train
        self.test = test
        self.val = val
        self.lr = lr
        self.momentum = 0.9
        self.log_path = log_path
        self.log_after = log_after
        self.model_path = model_path
        self.model_name = model_name
        self.val_accuracies = []
        self.criterion = None
        self.full_log_path = None
        self.scheduler = None
        self.optimizer = None

    def log_stats(self, epoch=0, iteration=0, running_loss=0.0):
        loss_record = '[%d, %5d] loss: %.3f' % \
            (epoch + 1, iteration + 1, running_loss / self.log_after)
        with open(self.full_log_path, 'a') as log:
            log.write(loss_record + '\n')
        print(loss_record)

    def validate(self, epoch=0, iteration=0):
        self.network.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_loss = 0.0
            for sample in self.val:
                images, labels = (sample['images'].cuda(),
                                  sample['labels'].cuda())
                outputs = self.network(images)
                total_loss += self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            self.val_accuracies.append(100*correct/total)
            acc_record = '[%d, %5d] validation accuracy: %.3f' % \
                (epoch + 1, iteration + 1, 100*correct/total)
            print(acc_record)
            loss_record = '[%d, %5d] validation loss: %.3f' % \
                (epoch + 1, iteration + 1, total_loss)
            print(loss_record)

            with open(self.full_log_path, 'a') as log:
                log.write(acc_record + '\n')
                log.write(loss_record + '\n')

        return total_loss

    def train_and_validate(self):
        '''Given a network, train and validate on the provided data.

        This method performs training on the network with the provided data.
        This will persist the model to a file with a name derived from the
        parameters, and will aditionally log out to a file named based on the
        network parameters.

        Returns
        -------
        network: torch.nn.Module
        '''

        self.network.cuda()
        self.criterion = nn.CrossEntropyLoss()
        if self.mode.lower() == 'sgd':
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr,
                                       momentum=self.momentum)
        elif self.mode.lower() == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                         'min')
        if self.model_name is None:
            self.model_name = f'{self.mode}_e_{self.epochs}_lr_{self.lr:.3f}'
            if self.mode == 'sgd':
                self.model_name += f'_m_{self.momentum}'

        # create or overwrite the file
        self.full_log_path = os.path.join(self.log_path, self.model_name)
        log = open(self.full_log_path, 'w')
        log.close()

        for epoch in range(self.epochs):
            running_loss = 0.0

            # train
            self.network.train()
            for i, sample in enumerate(self.train):
                images, labels = (sample['images'].cuda(),
                                  sample['labels'].cuda())
                self.optimizer.zero_grad()
                outputs = self.network(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if self.log_after and (i + 1) % self.log_after == 0:
                    self.log_stats(epoch, i, running_loss)
                    running_loss = 0

            # validate
            val_loss = self.validate(epoch, i)
            scheduler.step(val_loss)
        plt.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies)
        plt.xlabel('number of epochs')
        plt.ylabel('percent accuracy')
        plt.show()

        print('done')

        network_path = os.path.join(self.model_path, self.model_name)
        torch.save(self.network.state_dict(), network_path)
        print(f'network saved to {network_path}')
