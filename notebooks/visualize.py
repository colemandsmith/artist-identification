import common_utils
import numpy as np
import torch.utils.data
from PIL import Image


def generate_random_image(size=224):
    return Image.fromarray(
        np.uint8(np.random.uniform(0, 255, (size, size, 3)))
    )


def maximize_random_img_for_artist(model, artist, lr=0.1, img=None,
                                   img_size=224, max_optim_steps=40,
                                   epsilon=0.1):
    # If not only done, freeze the model
    label_to_artist, artist_to_label = common_utils.init_label_dicts()
    label = artist_to_label[artist]
    model.cuda()
    model = model.eval()
    if not img:
        img = generate_random_image()

    img_var = torch.tensor(img.reshape(3, 224, 224)[None],
                           device='cuda').float().requires_grad_()
    optimizer = torch.optim.Adam([img_var], lr=lr)
    for n in range(max_optim_steps):
        print(n)
        optimizer.zero_grad()
        outputs = model(img_var)
        print(torch.max(outputs.data, 1))
        loss = -outputs[label]
        loss.backward()
        optimizer.step()

    return img_var
