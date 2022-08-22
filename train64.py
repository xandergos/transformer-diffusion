import argparse

import torch
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

import diffusion
import wandb
from swin_diffuser import SwinGenerator


def preprocess(x):
    x, = x
    x = T.Compose([
        T.ToTensor(),
        T.Resize(64),
        T.CenterCrop(64)
    ])(x)
    return x * 2 - 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fp', type=str,
                        help='file path for the dataset')
    parser.add_argument('--weights', type=str, default=None,
                        help='model weights for fine tuning')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train for (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--examples', type=int, default=16,
                        help='number of examples to log (default: 16)')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='logging interval (default: 1000)')
    parser.add_argument('--workers', type=int, default=4,
                        help='workers for dataset loading (default: 4)')
    parser.add_argument('--beta-schedule', type=str, default='linear',
                        help='DDPM beta schedule (default: linear)')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='number of diffusion timesteps (default: 1000)')
    parser.add_argument('-c', '--base-channels', type=int, default=128,
                        help='base number of chanels (default: 128)')
    parser.add_argument('-d', '--decoder-depth', type=int, default=3,
                        help='decoder depth (default: 3)')
    parser.add_argument('-e', '--backbone-dims', type=int, default=96,
                        help='patch embed dims for swin transformer (default: 96)')
    parser.add_argument('-w', '--window-size', type=int, default=16,
                        help='window size for swin transformer (default: 16)')
    parser.add_argument('-t', '--time-dims', type=int, default=64,
                        help='embedding dimension for time vector (default: 64)')
    args = parser.parse_args()

    wandb.init(project='TDDPM', config={k: v for k, v in args.__dict__.items() if k not in ['fp', 'log_interval', 'workers', 'epochs', 'examples', 'weights']})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    schedule = diffusion.linear_beta_schedule(args.timesteps)
    if args.beta_schedule.lower() == 'linear':
        pass
    elif args.beta_schedule.lower() == 'cosine':
        schedule = diffusion.cosine_beta_schedule(args.timesteps)
    else:
        print("Unknown beta schedule. Options are linear or cosine.")

    diff_utils = diffusion.DiffusionUtils(schedule)

    m = SwinGenerator(args.base_channels, args.decoder_depth, args.backbone_dims,
                      args.window_size, args.time_dims).to(device)
    if args.weights is not None:
        m.load_state_dict(torch.load(args.weights))
    wandb.watch(m, log_freq=args.log_interval)
    optimizer = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

    lsun_bedroom = wds.WebDataset(args.fp).shuffle(1000).decode("rgb").to_tuple("input.jpg").map(preprocess)
    dl = DataLoader(lsun_bedroom, batch_size=args.batch_size, num_workers=args.workers)

    loss_sum = 0
    log_step = 0
    for epoch in range(args.epochs):
        epoch_loss_sum = 0
        epoch_step = 0
        pbar = tqdm(enumerate(dl))
        for step, batch in pbar:
            if batch.shape[0] != args.batch_size:
                continue

            optimizer.zero_grad()

            batch = batch.to(device)

            t = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long()
            loss = diff_utils.p_losses(m, batch, t, loss_type="l2")
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})

            loss_sum += loss.item()
            log_step += 1
            epoch_loss_sum += loss.item()
            epoch_step += 1

            with torch.no_grad():
                if log_step == args.log_interval:
                    torch.random.manual_seed(3117)
                    samples = diff_utils.sample(m, args.examples, device, 64).cpu() * .5 + .5
                    torch.save(m.state_dict(), 'model64.pt')

                    torch.random.seed()

                    wandb.log({"loss": loss_sum / log_step, "examples": [wandb.Image(im) for im in samples],
                               "epoch": epoch})
                    loss_sum = 0
                    log_step = 0

        scheduler.step(epoch_loss_sum / epoch_step)

    print('Training complete.')
