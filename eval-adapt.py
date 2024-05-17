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


def nearest_snr(snr):
    min_diff = 1000
    snr_int = 0
    for snr_ls in [0, 5, 10, 15, 20]:
        if abs(snr_ls - snr) < min_diff:
            min_diff = abs(snr_ls - snr)
            snr_int = snr_ls
    return snr_int


def inspect_noise(model: DeepJSCC):
    chan = model.channel
    blank = torch.ones(64, 512, 1, 1)
    c_b = chan(blank)
    idxs_0 = torch.arange(256) * 2
    idxs_1 = idxs_0 + 1
    chn = c_b.reshape(64, -1)
    chn_0 = c_b[:, idxs_0].reshape(64, -1)
    chn_1 = c_b[:, idxs_1].reshape(64, -1)
    c_max = torch.max(torch.abs(chn - 1))
    c_var = torch.var(chn)
    # print(c_var, c_max, c_max/torch.sqrt(c_var))
    # print(torch.var(chn_0, dim=1).sum() / 64)
    if (torch.mean(chn_0, 1) - torch.mean(chn_1, 1)).abs().sum() > 28:
        c_var_2 = (torch.var(chn_0, dim=1).sum() / 64 + torch.var(chn_1, dim=1).sum() / 64) / 2
        snr = 10 * torch.log10(1 / c_var_2)
        return 'Rician', nearest_snr(snr)
    elif c_max / torch.sqrt(c_var) > 10:
        c_var_2 = torch.var(chn[torch.abs(chn - 1) < c_max / 6])
        snr = 10 * torch.log10(1 / c_var_2)
        return 'Impulse', nearest_snr(snr)
    else:
        snr = 10 * torch.log10(1 / c_var)
        return 'AWGN', nearest_snr(snr)


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default='AWGN', type=str, help='channel type')
    return parser.parse_args()


def main():
    args = config_parser()
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                    download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64)

    c = 512
    print(f'c={c}')
    criterion = nn.MSELoss(reduction='mean').cuda()
    model = DeepJSCC(c=c, channel_type=args.channel, snr=0)
    # model.load_state_dict(torch.load(args.saved, map_location=torch.device('cuda:0')))
    model.cuda()
    psnr_list = []
    for snr in [0, 5, 10, 15, 20]:
        model.change_channel(args.channel, snr)
        chan_type, chan_snr = inspect_noise(model)
        model.load_state_dict(torch.load(f'./saved/cifar10_1000_0.17_{chan_snr}.00_64_{chan_type}_512.pth', map_location=torch.device('cuda:0')))
        model.eval()
        test_mse = torch.tensor(0.0)
        for images, _ in tqdm((test_loader), leave=False, disable=True):
            images = images.cuda()
            outputs = model(images)
            images = image_normalization('denormalization')(images)
            outputs = image_normalization('denormalization')(outputs)
            loss = criterion(outputs, images)
            test_mse += loss.item()
        psnr = 10 * torch.log10(255 ** 2 / (test_mse / len(test_loader)))
        psnr_list.append(psnr)
    print(f'psnr of adaptive model on {args.channel} is {psnr_list}')


if __name__ == '__main__':
    main()
