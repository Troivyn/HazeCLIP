import os
import torch

from tqdm import trange
from omegaconf import OmegaConf
from argparse import ArgumentParser

from modules.optimizer import Lion
from modules.MSBDN import MSBDN
from modules.loss import CharbonnierLoss
from modules.datasets import HazeOnlineDataset
from modules.loader import MultiEpochsDataLoader


def main(args):
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Define the network --- #
    model = MSBDN()

    # --- Build optimizer --- #
    optimizer = Lion(model.parameters(), lr=cfg.train.lr, weight_decay=1e-3, betas=(0.9, 0.99))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs, eta_min=1e-7)

    # --- Multi-GPU --- #
    model.to(device)

    # --- Load the network weight --- #
    os.makedirs(cfg.train.exp_dir + 'checkpoints/', exist_ok=True)

    print('--- Hyper-parameters for training ---')
    print('learning_rate: {}\ntrain_batch_size: {}\nnum_epochs: {}'.format(cfg.train.lr, cfg.train.batch_size, cfg.train.num_epochs))

    # --- Load training data and validation/test data --- #
    train_dataset = HazeOnlineDataset(cfg.data.gt_folder, cfg.data.depth_folder, cfg.data.crop_size)
    train_data_loader = MultiEpochsDataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=24, drop_last=True, pin_memory=True)

    loss_fn = CharbonnierLoss()

    for epoch in trange(cfg.train.num_epochs):
        model.train()
        torch.cuda.empty_cache()

        for train_data in train_data_loader:
            haze, gt, _ = train_data
            haze = haze.to(device)
            gt = gt.to(device)

            # --- Zero the parameter gradients --- #
            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            dehaze = model(haze)

            loss = loss_fn(dehaze, gt)
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        # --- Save the network parameters --- #
        if epoch % 50 == 49:
            torch.save(model.state_dict(), f'{cfg.train.exp_dir}checkpoints/epoch{epoch + 1}.pth')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
