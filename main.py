import argparse
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from project import Project
import requests
import time
import torch
from torch import multiprocessing
from torch import nn
from torch import optim
from torchvision import transforms, models
import traceback
from utils import pt_util


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='<~Style Transfer!~>')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
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


def load_image(img_path, max_size=400, shape=None) -> torch.Tensor:
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


def to_ndarray(tensor: torch.Tensor) -> np.ndarray:
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
    print(array[:, :, 0])
    array = array * np.array((0.229, 0.224, 0.225)) + np.array((0.484, 0.456, 0.406))
    print(array[:, :, 0])
    array = array.clip(0, 1)
    return array


def main():
    # args = get_args()
    # model = load_model()
    # print('VGG19:\n', model)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using Device:', device)
    # model.to(device)

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

    plt.show()


if __name__ == '__main__':
    main()
    # Data Transforms and Datasets

    # # Train on GPU (if CUDA is available)
    # use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda' if use_cuda else 'cpu')
    # print('Using Device', device)


    # # Create Optimizer
    # opt = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # num_workers = multiprocessing.cpu_count()
    # print('Number of CPUs:', num_workers)
    # kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    # # replace with get_dataloaders() in the template overall
    # train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = DataLoader(data_test, batch_size=args.test_batch_size, **kwargs)

    # start_epoch = network.load_last_model(str(project.WEIGHTS_PATH))
    # train_losses, test_losses, test_accuracies = pt_util.read_log(str(project.LOG_PATH), ([], [], []))
    # test_loss, test_accuracy = test(network, device, test_loader)
    # test_losses.append((start_epoch, test_loss))
    # test_accuracies.append((start_epoch, test_accuracy))

    # try:
        # for epoch in range(start_epoch, args.epochs + 1):
            # train_loss = train(network, device, train_loader, opt, epoch, args.print_interval)
            # test_loss, test_accuracy = test(network, device, test_loader)
            # train_losses.append((epoch, train_loss))
            # test_losses.append((epoch, test_loss))
            # test_accuracies.append((epoch, test_accuracy))
            # pt_util.write_log(str(project.LOG_PATH), (train_losses, test_losses, test_accuracies))
            # network.save_best_model(test_accuracy, str(project.WEIGHTS_PATH) + '/%03d.pt' % epoch)
    # except KeyboardInterrupt as ke:
        # print('Manually interrupted execution...')
    # except:
        # traceback.print_exc()
    # finally:
        # # TODO: Shouldn't this be saved to most recent epoch
        # print('Saving model in its current state')
        # network.save_model(str(project.WEIGHTS_PATH) + '/%03d.pt' % epoch, 0)
        # ep, val = zip(*train_losses)
        # pt_util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
        # ep, val = zip(*test_losses)
        # pt_util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
        # ep, val = zip(*test_accuracies)
        # pt_util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')

