# to be implemented
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from utils import get_psnr, image_normalization
import os
from model import DeepJSCC
torch.manual_seed(1)

def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default='AWGN', type=str, help='channel type')
    parser.add_argument('--saved', type=str, help='saved_path')
    parser.add_argument('--snr', default=20, type=int, help='snr')
    parser.add_argument('--test_image', default='./demo/kodim08.png', type=str, help='demo_image')
    parser.add_argument('--times', default=10, type=int, help='num_workers')
    return parser.parse_args()


def main():
    args = config_parser()
    transform = transforms.Compose([transforms.ToTensor()])
    test_image = Image.open(args.test_image)
    test_image.load()
    test_image = transform(test_image)

    file_name = os.path.basename(args.saved)
    c = file_name.split('_')[-1].split('.') [0]
    c = int(c)
    print(f'c={c}')
    model = DeepJSCC(c=c, channel_type=args.channel, snr=args.snr)
    # model.load_state_dict(torch.load(args.saved))
    model.load_state_dict(torch.load(args.saved,map_location=torch.device('cuda:0')))
    model.change_channel(args.channel, args.snr)

    psnr_all = 0.0

    for i in range(args.times):
        demo_image = model(test_image)
        demo_image = image_normalization('denormalization')(demo_image)
        gt = image_normalization('denormalization')(test_image)
        psnr_all += get_psnr(demo_image, gt)
    demo_image = image_normalization('normalization')(demo_image)
    demo_image = torch.cat([test_image, demo_image], dim=1)
    demo_image = transforms.ToPILImage()(demo_image)
    demo_image.save('./run/{}_{}_{}'.format(args.saved.split('/')[-1], args.snr, args.test_image.split('/')[-1]))
    print("psnr on {} is {}".format(args.test_image, psnr_all / args.times))


if __name__ == '__main__':
    main()
