import numpy as np
from PIL import Image
import copy
import torch
from torch.autograd import Variable

import art_identify.common_utils as common_utils


def generate_random_image(size=224):
    return Image.fromarray(
        np.uint8(np.random.uniform(110, 190, (size, size, 3)))
    )


def preprocess(image):
    """Normalizes the image and package as pytorch Variable.

    Normalization is done based off imagenet statistics.

    Parameters
    ----------
    image: np.array

    Returns
    -------
    torch.autograd.Variable
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im_arr = np.float32(image)
    im_arr = im_arr.transpose(2, 0, 1)

    im_arr /= 255
    for channel in range(len(im_arr)):
        im_arr[channel] -= mean[channel]
        im_arr[channel] /= std[channel]
    im_tensor = torch.from_numpy(im_arr).cuda().float()

    im_tensor.unsqueeze_(0)
    im_var = Variable(im_tensor, requires_grad=True)
    return im_var


def denormalize(image):
    """Denormalizes the image based on imagenet statistics.

    Parameters
    ----------
    image: torch.autograd.Variable

    Returns
    -------
    np.array
    """
    r_mean = [-0.485, -0.456, -0.406]
    r_std = [1/0.229, 1/0.224, 1/0.225]
    recreated = copy.copy(image.numpy()[0])
    for channel in range(len(recreated)):
        recreated[channel] /= r_std[channel]
        recreated[channel] -= r_mean[channel]
    recreated = np.round(recreated * 255)
    recreated = np.uint8(recreated).transpose(1, 2, 0)
    return recreated


def maximize_img_for_artist(model, artist, lr=0.1, img=None,
                            img_size=224):
    """Maximizes an image for the artist from the model.

    Parameters
    ----------
    model: torch.nn.Module
    artist: str
    lr: float
    img: np.array
    img_size: int
    """
    # If not only done, freeze the model
    label_to_artist, artist_to_label = common_utils.init_label_dicts()
    label = artist_to_label[artist]
    model.cuda()
    model = model.eval()
    if not img:
        img = generate_random_image(img_size)

    img_var = preprocess(img)
    optimizer = torch.optim.Adam([img_var], lr=0.5, weight_decay=1e-2)
    for n in range(500):
        optimizer.zero_grad()
        outputs = model(img_var)
        loss = -outputs[0, label]
        model.zero_grad()
        loss.backward()
        if n == 0 or n % 50 == 0:
            print(loss)
        optimizer.step()
    img = img_var.detach().cpu()

    return img_var
