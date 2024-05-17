# to be implemented
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets
from utils import get_psnr, image_normalization
import os
from model import DeepJSCC
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.manual_seed(1)

def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default='AWGN', type=str, help='channel type')
    parser.add_argument('--saved', type=str, help='saved_path')
    return parser.parse_args()


def main():
    args = config_parser()
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                    download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64)

    file_name = os.path.basename(args.saved)
    c = file_name.split('_')[-1].split('.') [0]
    c = int(c)
    print(f'c={c}')
    criterion = nn.MSELoss(reduction='mean').cuda()
    model = DeepJSCC(c=c, channel_type=args.channel, snr=0)
    model.load_state_dict(torch.load(args.saved, map_location=torch.device('cuda:0')))
    model.cuda()
    psnr_list = []
    for snr in [0, 5, 10, 15, 20]:
        model.change_channel(args.channel, snr)
        model.eval()
        test_mse = torch.tensor(0.0)
        for images, _ in tqdm((test_loader), leave=False, disable=True):
            images = images.cuda()
            outputs = model(images)
            images = image_normalization('denormalization')(images)
            outputs = image_normalization('denormalization')(outputs)
            loss = criterion(outputs, images)
            test_mse += loss.item()
        psnr = 10 * torch.log10(255 ** 2 / (test_mse /len(test_loader)))
        psnr_list.append(psnr)
    print(f'psnr of {file_name} on {args.channel} is {psnr_list}')


if __name__ == '__main__':
    main()
