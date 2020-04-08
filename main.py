#!/usr/bin/env python
import argparse
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from paths import ProjectPaths
import requests
import time
import torch
from torch import multiprocessing
from torch import nn
from torch import optim
from torchvision import transforms, models
import traceback
from utils import pt_util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='<~Style Transfer!~>')
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--print_interval', type=int, default=10)
    return parser.parse_args()


def load_model() -> nn.Module:
    """
    Returns the pre-trained features of deep neural network architecture trained on ImageNet task.

    Arguments:
        architecture: str, desired architecture (vgg19, resnet101, googlenet, etc.).

    Returns:
        model: nn.Module, pre-trained model from torchvision of specified architecture
    """
    vgg19 = models.vgg19(pretrained=True).features
    for parameter in vgg19.parameters():
        parameter.requires_grad = False
    return vgg19


def load_image(img_path, max_size=400, shape=None):
    """
    Converts an image at the specified path to a torch.Tensor object

    Arguments:
        img_path: str, path to an image. This can either be a valid file path or URL.
        max_size: int, the maximum permitted size of an image in both the vertical and horizontal
            dimensions.
        shape: tuple, specifying the shape of a given image (Optional).

    Returns:
        (tensor_img, native_img_shape): (torch.Tensor, tuple), image at the specified link in tensor format. Convolutions
        process tensors in the shape (N=batch_size, N=channels_in, H=image_height, W=image_width) so
        we return the tensor in this format and the shape of the original PIL image.
    """
    if 'http' in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    image_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # shape before torch indexing is (3, H, W)
    tensor_img = image_transform(image)[:3, :, :].unsqueeze(0)
    return tensor_img, tensor_img.size()[2:]


def to_ndarray(tensor) -> np.ndarray:
    """
    Convert a torch.Tensor object into an np.ndarray object

    Arguments:
        tensor: torch.Tensor

    Returns:
        array: np.ndarray
    """
    # make a copy and move it to the cpu
    array = tensor.clone().to('cpu').detach()
    # get rid of batch dimension
    array = array.numpy().squeeze()
    # restore the original shape of the image, undo channels arrangement
    array = array.transpose(1, 2, 0)
    array = array * np.array((0.229, 0.224, 0.225)) + np.array((0.484, 0.456, 0.406))
    array = array.clip(0, 1)
    return array


def get_features(model: nn.Module, tensor_img) -> dict:
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = tensor_img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def get_gramian(tensor):
    """
    <description>

    Arguments:
        tensor: torch.Tensor,

    Returns:
        gramian: torch.Tensor,
    """
    _, input_channels, height, width = tensor.size()
    tensor = tensor.view(input_channels, height * width)
    gramian = tensor @ tensor.T
    return gramian


def main():
    args = parse_args()
    print(args)
    model = load_model()
    paths = ProjectPaths()
    print('VGG19:\n', model)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using Device:', device)
    model.to(device)

    content_path = "https://pbs.twimg.com/profile_images/1228153568688918529/au9ifIiK_400x400.jpg"
    content_img, content_shape = load_image(content_path)
    print(content_img.size(), content_shape)
    style_path = "https://www.1st-art-gallery.com/frame-preview/4578922.jpg?sku=Unframed&thumb=0&huge=1"
    style_img, _ = load_image(style_path, shape=content_shape)

    content_img.to(device)
    style_img.to(device)

    content_display = to_ndarray(content_img)
    style_display = to_ndarray(style_img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(content_display)
    ax2.imshow(style_display)
    fig.savefig(str(paths.IMAGES_PATH) + '/default.png')

    content_features = get_features(model, content_img)
    style_features = get_features(model, style_img)

    style_gramians = {layer: get_gramian(style_features[layer]) for layer in style_features}

    target_img = content_img.clone().to(device)
    target_img.requires_grad = True

    style_weights = {
        'conv1_1': 1.,
        'conv2_1': 0.75,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2
    }

    content_weight = 1  # alpha
    style_weight = 1e6  # beta

    optimizer = optim.Adam([target_img], lr=args.lr)

    for ii in range(args.iterations + 1):
        optimizer.zero_grad()
        target_features = get_features(model, target_img)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # TODO: Put this into a method
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gramian = get_gramian(target_feature)
            _, input_channels, height, width = target_feature.size()
            style_gramian = style_gramians[layer]
            curr_layer_loss = style_weights[layer] * torch.mean((target_gramian - style_gramian)**2)
            style_loss += curr_layer_loss / (input_channels * height * width)
        loss = content_weight * content_loss + style_weight * style_loss
        loss.backward()
        optimizer.step()

        if ii % args.print_interval == 0:
            print("Iteration:", ii)
            print("Loss:", loss.item())
            curr_path = str(paths.IMAGES_PATH) + '/result_step_%03d.png' % ii
            plt.imsave(curr_path, to_ndarray(target_img))

    # Show final figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(to_ndarray(content_img))
    ax2.imshow(to_ndarray(target_img))
    fig.savefig(str(paths.IMAGES_PATH) + '/final.png')


if __name__ == '__main__':
    main()

