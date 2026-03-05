
import os
import sys
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion_planner import DiffusionPlanner, normalize_data

# Import training components from diffusers
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler


class Planning2DDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, action_horizon: int = 64):
        self.dataset_path = dataset_path.rstrip('/')
        self.stats = {'min': -5., 'max': 5.}
        self.file_indices = {}
        idx = 0
        traj_files = []
        for f in os.listdir(self.dataset_path):
            # Only load trajectory arrays, not other .npy artifacts
            if re.match(r'^maze_\d+_traj_\d+\.npy$', f):
                traj_files.append(f)
        # deterministic order
        traj_files.sort(key=lambda s: (int(s.split('_')[1]), int(s.split('_')[3].split('.')[0])))
        for f in traj_files:
            self.file_indices[idx] = f
            idx += 1
        self.file_len = idx
        self.indices = self._make_indices(action_horizon)

    def _make_indices(self, horizon):
        indices = []
        for file_idx in range(self.file_len):
            fname = self.file_indices[file_idx]
            traj = np.load(os.path.join(self.dataset_path, fname))
            path_length = traj.shape[1]
            if path_length < horizon:
                continue
            max_start = path_length - horizon
            for start in range(max_start + 1):
                indices.append((file_idx, start, start + horizon))
        return np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, start, end = self.indices[idx]
        traj_fname = self.file_indices[file_idx]
        traj_array = np.load(os.path.join(self.dataset_path, traj_fname))[:, start:end].T.astype(np.float32)
        nsample = {}
        nsample['action'] = self._norm(traj_array)
        nsample['start'] = nsample['action'][0:1]
        nsample['goal'] = nsample['action'][-1:]
        image_num = traj_fname.split('_')[1]
        img_path = os.path.join(self.dataset_path, f'maze_occu_{image_num}.png')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Missing maze image for trajectory {traj_fname}: {img_path}")
        # cv2 loads BGR; convert to RGB and normalize to [0,1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.moveaxis(img, -1, 0)  # (3,H,W)
        nsample['image'] = np.expand_dims(img, axis=0).astype(np.float32)
        return nsample

    def _norm(self, x):
        x = (x - self.stats['min']) / (self.stats['max'] - self.stats['min'])
        return (x * 2 - 1).astype(np.float32)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    dataset = Planning2DDataset(args.data_dir, action_horizon=args.action_horizon)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=min(8, os.cpu_count() or 1), pin_memory=True
    )

    planner = DiffusionPlanner(device=device)
    nets = planner.nets
    noise_scheduler = planner.noise_scheduler
    obs_horizon = 1
    noise_pred_net = nets['noise_pred_net']

    ema = EMAModel(model=nets, power=0.75)
    optimizer = torch.optim.AdamW(nets.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=500,
                                 num_training_steps=len(dataloader) * args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f'Training on {len(dataset)} samples, {args.epochs} epochs')

    for epoch in range(args.epochs):
        epoch_loss = []
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            nimage = batch['image'][:, :obs_horizon].to(device)
            naction = batch['action'].to(device)
            nstart = batch['start'].to(device)
            ngoal = batch['goal'].to(device)

            image_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
            image_features = image_features.reshape(*nimage.shape[:2], -1)
            obs_features = torch.cat([image_features, nstart, ngoal], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)

            noise = torch.randn_like(naction, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (naction.shape[0],), device=device).long()
            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

            noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step(nets)
            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        print(f'Epoch {epoch+1} loss: {avg_loss:.6f}')
        if (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f'dipper_epoch{epoch+1}.pth')
            torch.save(ema.averaged_model.state_dict(), ckpt)
            print(f'Saved {ckpt}')

    final_ckpt = os.path.join(args.checkpoint_dir, 'dipper_final.pth')
    torch.save(ema.averaged_model.state_dict(), final_ckpt)
    print(f'Saved final model to {final_ckpt}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None,
                        help='Path to DiPPeR dataset (default: ../data/data_20230908-113959_10k_maps_100_traj)')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--action-horizon', type=int, default=64)
    parser.add_argument('--save-every', type=int, default=10)
    args = parser.parse_args()

    args.data_dir = args.data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'data_20230908-113959_10k_maps_100_traj')
    if not os.path.exists(args.data_dir):
        print(f'Data dir not found: {args.data_dir}')
        print('Download dataset and extract to data/data_20230908-113959_10k_maps_100_traj/')
        sys.exit(1)

    train(args)


if __name__ == '__main__':
    main()
