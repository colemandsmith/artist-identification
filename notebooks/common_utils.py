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


def init_label_dicts():
    label_to_artist = {}
    artist_to_label = {}
    label_counter = 0

    with open(FILTERED, "r", encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, quotechar='\"')
        next(reader)
        for row in reader:
            if row[0] not in artist_to_label:
                label_to_artist[label_counter] = row[0]
                artist_to_label[row[0]] = label_counter
                label_counter += 1


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
To follow along with the paper, we are going to randomly crop 224x244
images out of the training images.
'''
class ArtistImageDataset(torch.utils.data.Dataset):
    def __init__(self, text_file, img_dir, transform=train_loader_transform):
        self.name_frame = pd.read_csv(text_file, sep=",",
                                      usecols=range(11, 12))
        self.label_frame = pd.read_csv(text_file, sep=",", usecols=range(1))
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
    print([label_to_artist[s.data.numpy()[()]] for s in sample['labels']])


def set_optimizer(network, mode, new_lr, momentum):
    if mode.lower() == 'sgd':
        return optim.SGD(network.parameters(), lr=new_lr, momentum=momentum)
    elif mode.lower() == 'adam':
        return optim.Adam(network.parameters(), lr=new_lr)
    return None


def train_with_params(network, mode, epochs, train, test, val, lr=1e-3,
                      momentum=0.9, log_after=80, log_path=LOG_DIR,
                      model_path=MODEL_DIR, model_name=None):
    '''Given a network, train on the provided data with the provided parameters.

    This method performs training on the network with the provided data. This
    will persist the model to a file with a name derived from the parameters,
    and will aditionally log out to a file named based on the network
    parameters.

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

    Returns
    -------
    network: torch.nn.Module
    '''

    network.cuda()
    lower_lr = lr / 10
    criterion = nn.CrossEntropyLoss()
    optimizer = set_optimizer(network, mode, lr, momentum)
    if model_name is None:
        model_name = '%s_e_%d_lr_%.3f' % (mode, epochs, lr)
        if mode == 'sgd':
            model_name += '_m_%d' % momentum

    log = open(os.path.join(log_path, model_name), 'w')

    val_acc = []
    iterations = []
    train_len = len(train)

    for epoch in range(epochs):
        running_loss = 0
        results_track = []

        for i, sample in enumerate(train):
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            optimizer.zero_grad()
            outputs = network(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if log_after and (i+1) % log_after == 0:
                loss_record = '[%d, %5d] loss: %.3f' % \
                    (epoch + 1, i + 1, running_loss / log_after)
                log.write(loss_record)
                print(loss_record)

                running_loss = 0.0
                with torch.no_grad():
                    total = 0
                    correct = 0
                    for sample in val:
                        images, labels = (sample['images'].cuda(),
                                          sample['labels'].cuda())
                        outputs = network(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    val_acc.append(100*correct/total)
                    iterations.append((i+1) + (train_len * epoch))
                    acc_record = '[%d, %5d] test accuracy: %.3f' % \
                        (epoch + 1, i + 1, 100*correct/total)
                    log.write(acc_record)
                    print(acc_record)
                    # if we haven't sufficiently improved, decrease the
                    # learning rate
                    if len(results_track) > 4:
                        if results_track[-1] - results_track[0] < 1:
                            lr = lower_lr
                            optimizer = set_optimizer(mode, lr, momentum)
                            lr_record = '[%d, %5d] new learning rate: %3f' % lr
                            log.write(lr_record)
                            print(lr_record)
                        # shift this to be a moving
                        results_track.pop(0)
                        results_track.append(val_acc[-1])
                    else:
                        results_track.append(val_acc[-1])
    plt.plot(iterations, val_acc)
    plt.xlabel('number of iterations')
    plt.ylabel('percent accuracy')
    plt.show()

    print('done')

    log.close()

    network_path = os.path.join(model_path, model_name)
    torch.save(network.state_dict(), network_path)
    print(f'network saved to {network_path}')

    return network
