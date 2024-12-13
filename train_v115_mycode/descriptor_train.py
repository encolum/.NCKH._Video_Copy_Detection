import logging
import sys, os
import argparse
import random
# import math
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import torch.nn.functional as F
import numpy as np
# import torch.nn as nn

from mmcv import Config
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
# from modeling import EMA
from model_factory.utils import build_model, build_dataset

print('Importing modules done')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--t', type=float, default=0.05)
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--work_dir', type=str, default='')
    # parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--clip_grad_norm', type=float, default=1)
    parser.add_argument('--local_rank', type=int, default=0)  
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--instance_mask', action='store_true', default=False)
    parser.add_argument('--entropy_loss', action='store_true', default=False)
    parser.add_argument('--do_ema', action='store_true', default=False)
    parser.add_argument('--do_fgm', action='store_true', default=False)
    parser.add_argument('--entropy_weight', type=float, default=30)
    parser.add_argument('--ici_weight', type=float, default=1.)
    parser.add_argument('--fp16', action='store_true', default=False)  
    parser.add_argument('--checkpointing', action='store_true', default=False)
    parser.add_argument('--concat_dataset', action='store_true', default=False)
    parser.add_argument('--product_loss', action='store_true', default=False)

    args = parser.parse_args()
    return args


args = parse_args()

work_dir = r"D:\learn\giaotrinh\new\video similarity\mini_train_v115"
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
print_freq = args.print_freq
# resume = args.resume if args.resume != '' else None
warmup_ratio = args.warmup_ratio


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(args.seed)


def all_gather(local_rank, world_size, **tensors):
    tensors = list(tensors.values())
    _dims = [t.shape[-1] for t in tensors]
    tensors = torch.cat(tensors, dim=-1)
    tensors_all = [torch.zeros_like(tensors) for _ in range(world_size)]
    tensors_all[local_rank] = tensors
    tensors_all = torch.cat(tensors_all, dim=0)

    results = list()
    dimStart = 0
    assert sum(_dims) == tensors_all.shape[-1]
    for d in _dims:
        results.append(tensors_all[..., dimStart: dimStart + d])
        dimStart += d

    return tuple(results)



if args.rank == 0:
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(f"{work_dir}/checkpoints", exist_ok=True) 
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(work_dir + '/log.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    

cfg = Config.fromfile(r"D:\learn\giaotrinh\new\video similarity\mini_train_v115\config_v115.py")
cfg.local_rank = args.local_rank


train_dataset = build_dataset(cfg.data.train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)


model = build_model(cfg.model)
model.cuda()

ema = EMA(model, 0.999)
ema.register()

opt = AdamW(model.parameters(), lr=lr)
batch_size = batch_size  
stepsize = (len(train_dataset) // batch_size + 1)
total_steps = (len(train_dataset) // batch_size + 1) * epochs
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_ratio * total_steps,
                                            num_training_steps=total_steps)


start_epoch = 0


def contrast_loss_fn(emb_a, emb_b, temperature, mask, m=None):
    bz = emb_a.size(0)
    emb = torch.cat([emb_a, emb_b], dim=0)  # 2xbz
    sims = emb @ emb.t()
    diag = torch.eye(sims.size(0)).to(sims.device)

    small_value = torch.tensor(-10000.).to(sims.device).to(sims.dtype)
    sims = torch.where(diag.eq(0), sims, small_value)
    gt = torch.cat([torch.arange(bz) + bz, torch.arange(bz)], dim=0).to(sims.device)
    mask = torch.cat([mask, mask], dim=0).bool()

    if args.margin > 0:
        loss_ = F.cross_entropy((sims - diag * args.margin) / temperature, gt, reduction="none")[mask.bool()].mean()
    else:
        loss_ = F.cross_entropy(sims / temperature, gt, reduction="none")[mask.bool()].mean()

    return loss_


def entropy_loss_fn(sims, mask):
    device = sims.device
    
    diag = torch.eye(sims.size(0)).to(device)
    local_mask = (1 - diag)
    small_value = torch.tensor(-10000.).to(device).to(sims.dtype)
    max_non_match_sim = torch.where(local_mask.bool(), sims, small_value)[mask.bool()].max(dim=1)[0]
    closest_distance = (1 / 2 - max_non_match_sim / 2).clamp(min=1e-6).sqrt()
    entropy_loss_ = -closest_distance.log().mean() * args.entropy_weight
    return entropy_loss_


def train_step(batch_data):
    # Move video data to GPU if it's not already there
    vid_a, vid_b = batch_data["vid_a"].cuda(), batch_data["vid_b"].cuda()
    bz = batch_data["img_a"].size(0)
    device = batch_data["img_a"].device  # Ensure device is set properly

    # Concatenate and process embeddings on GPU
    cat_x = torch.cat([batch_data["img_a"], batch_data["img_b"]], dim=0).to(device)
    embeds = model(x=cat_x).to(device)  # Ensure embeddings are on GPU
    embeds_norm = embeds / embeds.norm(dim=1, keepdim=True)
    emb_a, emb_b = embeds[:bz], embeds[bz:2 * bz]
    emb_a_norm, emb_b_norm = embeds_norm[:bz], embeds_norm[bz:2 * bz]

    # Ensure all_gather output is also on GPU
    ga_emb_a_norm, ga_emb_b_norm, ga_vid_a, ga_vid_b = all_gather(
        args.rank, 1, emb_a=emb_a_norm, emb_b=emb_b_norm, 
        vid_a=vid_a[..., None], vid_b=vid_b[..., None]
    )

    # Ensure sims_norm is computed on the GPU
    sims_norm = ga_emb_a_norm @ ga_emb_b_norm.t()

    # Create a rank mask on the GPU
    rank_mask = torch.zeros(bz, device=device)
    rank_mask[args.rank * bz:(args.rank + 1) * bz] = 1

    # Calculate loss on GPU
    if args.product_loss:
        match_sim = (emb_a_norm * emb_b_norm).sum(dim=1)
        entropy_loss_ = (1 - match_sim).exp().mean().to(device)
    else:
        entropy_loss_ = entropy_loss_fn(sims_norm, rank_mask)

    ici_loss_ = contrast_loss_fn(ga_emb_a_norm, ga_emb_b_norm, args.t, rank_mask) * args.ici_weight

    return ici_loss_, entropy_loss_

global_step = 0
for _e in range(start_epoch, epochs):
    print(f"Epoch: {_e}")
    print('Training model')
    model.train()
    for _b, batch in enumerate(train_loader):
        print(f"Batch: {_b}")
        for _k, _v in batch.items():
            if isinstance(_v, torch.Tensor):
                batch[_k] = _v.cuda()
        print('Zero_grad')
        # Removed FGM logic
        opt.zero_grad()
        if args.fp16:
            # Removed mixed precision training
            pass  # Handle training without fp16
        print("Zero_grad done")
        print('ici + entropy loss')
        ici_loss, entropy_loss = train_step(batch)
        print('ici + entropy loss done')
        loss = ici_loss + entropy_loss
        print('backpropagating loss')
        loss.backward()
        print('backpropagation done')
        # Removed FGM logic
        print('step')
        opt.step()
        if args.do_ema:
            ema.update()
        scheduler.step()
        print('step done')

        global_step += 1
        if args.rank == 0 and _b % print_freq == 0:
            logger.info('Epoch %d Batch %d Loss %.3f, ICI Loss %.3f, Entropy loss %.3f.' % (
                _e, _b, loss.item(), ici_loss.item(), entropy_loss.item())
            )

    if args.rank == 0:
        if args.do_ema:
            print('Doing ema')
            ema.apply_shadow()
        ckpt = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': _e}
        print('Saving checkpoint')
        torch.save(ckpt, work_dir + '/checkpoints/epoch_%d.pth' % _e)
        if args.do_ema:
            ema.restore()


