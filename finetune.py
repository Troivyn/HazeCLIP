import os
import torch

from tqdm import trange
from omegaconf import OmegaConf
from argparse import ArgumentParser

from CLIP import clip
from prompts import *
from modules.optimizer import Lion
from modules.MSBDN import MSBDN
from modules.datasets import TrainData
from modules.loader import MultiEpochsDataLoader
from clip_score import L_clip_from_feature, L_clip_MSE


def main(args):
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clip_model, _ = clip.load("CS-ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")  # ViT-B/32
    clip_model.to(device)
    for param in clip_model.parameters():
        param.requires_grad = False
    res_model, _ = clip.load("RN101", device=torch.device("cpu"), download_root="./clip_model/")
    res_model.to(device)
    for param in res_model.parameters():
        param.requires_grad = False

    clip_model.eval()
    res_model.eval()

    # --- Define the network --- #
    non_sky_features = clip.encode_text(clip_model, [non_sky_pos_prompts, non_sky_neg_prompts], device)
    sky_features = clip.encode_text(clip_model, [sky_pos_prompts, sky_neg_prompts], device)
    enhance_features = clip.encode_text(clip_model, enhance_prompts, device)

    model = MSBDN()

    # --- Build optimizer --- #
    optimizer = Lion(model.parameters(), lr=cfg.train.lr, weight_decay=1e-3, betas=(0.9, 0.99))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs, eta_min=1e-7)

    # --- Multi-GPU --- #
    model.to(device)
    model.load_state_dict(torch.load(cfg.train.resume, map_location=torch.device("cpu")))

    os.makedirs(cfg.train.exp_dir + 'checkpoints/', exist_ok=True)
    print('--- Hyper-parameters for training ---')
    print('learning_rate: {}\ntrain_batch_size: {}\nnum_epochs: {}'.format(cfg.train.lr, cfg.train.batch_size, cfg.train.num_epochs))

    # --- Load training data and validation/test data --- #
    train_dataset = TrainData(cfg.data.folder, cfg.data.crop_size)
    train_data_loader = MultiEpochsDataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=12, drop_last=True, pin_memory=True)

    id_loss_fn = L_clip_MSE()
    clip_loss_fn = L_clip_from_feature()

    for epoch in trange(cfg.train.num_epochs):
        model.train()
        torch.cuda.empty_cache()

        for train_data in train_data_loader:
            sky, non_sky, full = train_data
            sky = sky.to(device)
            non_sky = non_sky.to(device)
            full = full.to(device)

            # --- Zero the parameter gradients --- #
            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #

            sky = model(sky)
            non_sky = model(non_sky)
            full_ = model(full)

            identity_loss = id_loss_fn(res_model, full_, full, [1.0, 1.0, 1.0, 0.8, 0.3])

            sky_loss = clip_loss_fn(clip_model, sky, sky_features)
            desky_loss = clip_loss_fn(clip_model, non_sky, non_sky_features)
            enhance_loss = clip_loss_fn(clip_model, full_, enhance_features)

            loss = desky_loss + sky_loss + 0.5 * enhance_loss + 0.1 * identity_loss

            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        # --- Save the network parameters --- #
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), f'{cfg.train.exp_dir}checkpoints/epoch{epoch + 1}.pth')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
