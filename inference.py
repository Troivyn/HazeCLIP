import os
import glob
import torch
import torchvision

from PIL import Image
from tqdm import tqdm
from modules.MSBDN import MSBDN
from omegaconf import OmegaConf
from argparse import ArgumentParser
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode


def _convert_image_to_rgb(img):
    return img.convert("RGB")


transform = Compose([
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dehaze(model, image_path, folder):
    haze = Image.open(image_path)
    haze = transform(haze).unsqueeze(0).to(device)
    _, _, h, w = haze.shape
    haze = Resize((h // 16 * 16, w // 16 * 16), interpolation=InterpolationMode.BICUBIC, antialias=True)(haze)

    out = model(haze).squeeze(0)
    out = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(out)
    out = out.permute(1, 2, 0) * torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    out += torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()

    store_path = folder + os.path.basename(image_path)
    torchvision.utils.save_image(out.permute(2, 0, 1), store_path)


def main(args):
    cfg = OmegaConf.load(args.config)

    model = MSBDN()
    model = model.to(device)

    state_dict = torch.load(cfg.weight, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    os.makedirs(cfg.output_folder, exist_ok=True)
    images = glob.glob(f'{cfg.input_folder}*jpg')
    images += glob.glob(f'{cfg.input_folder}*png')

    with torch.no_grad():
        for image in tqdm(images):
            dehaze(model, image, cfg.output_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
