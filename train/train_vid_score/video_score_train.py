import logging
import sys, os
import argparse
import random
import pandas as pd
import torch
import torch.nn as nn
import logging
from vsc.baseline.model_factory.utils import build_dataset
import torch.nn.functional as F
from video.model import MS
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score
from mmcv import Config
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.tensorboard import SummaryWriter
import shutil
import tqdm
from torch.nn.utils import clip_grad_norm_





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=r"D:\VideoMatching\train\train_vid_score\config_for_video_score_train.py")  # Đường dẫn file cấu hình
    parser.add_argument('--batch_size', type=int, default=16)  # Kích thước batch
    parser.add_argument('--num_workers', type=int, default=0)  # Số worker để tải dữ liệu
    parser.add_argument('--feat_dim', type=int, default=512)  # Kích thước feature
    parser.add_argument('--bert_dim', type=int, default=768)  # Kích thước BERT embedding
    parser.add_argument('--output_dim', type=int, default=256)  # Kích thước đầu ra
    parser.add_argument('--bert_path', type=str, default=r"D:\VideoMatching\checkpoints_train_vid_score\googlebert-base")  # Đường dẫn pretrained BERT
    # parser.add_argument('--bert_path', type=str, default=r"D:\VideoMatching\checkpoints\chinese-roberta-wwm-ext-base")  # Đường dẫn pretrained BERT
    parser.add_argument('--val_ann_path', type=str, default=r"D:\VideoMatching\data\meta\train\train_ground_truth.csv")  # Đường dẫn annotation validation
    parser.add_argument('--max_frames', type=int, default=256)  # Số frame tối đa
    parser.add_argument('--lr', type=float, default=5e-5)  # Learning rate
    parser.add_argument('--print_freq', type=int, default=30)  # Tần suất in log
    parser.add_argument('--eval_freq', type=int, default=10)  # Tần suất đánh giá
    parser.add_argument('--work_dir', type=str, default=r"D:\VideoMatching\train\train_vid_score")  # Thư mục làm việc
    parser.add_argument('--resume', type=str, default=r"D:\VideoMatching\train\train_vid_score\checkpoints_new")  # Đường dẫn checkpoint để tiếp tục train
    parser.add_argument('--epochs', type=int, default=10)  # Số epoch
    parser.add_argument('--warmup_ratio', type=float, default=0.1)  # Tỷ lệ warmup
    parser.add_argument('--local_rank', type=int, default=0)  # Rank của process hiện tại
    parser.add_argument('--fp16', action='store_true', default=False)  # Sử dụng mixed precision
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)  # Sử dụng gradient checkpointing
    args = parser.parse_args()
    return args


logger = logging.getLogger()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1234)
#We need to build_dataset, model MS


# Hàm thực hiện một bước huấn luyện
def train_step(batch_data):
    labels = batch_data["labels"]
    frames = batch_data["frames"]
    model = MS(args)
    model.cuda()
    logits = model(frames)
    scores = logits.sigmoid()

    loss_ = F.binary_cross_entropy_with_logits(logits, labels.float())
    # scores = scores.squeeze(1)
    # labels = labels.squeeze(1)
    acc = (scores.round() == labels).float().mean().item()
    pn = (labels == 1).sum() / labels.size(0)

    ap_ = average_precision_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())

    return loss_, ap_, acc, pn


def main(args, cfg):
    
    
    work_dir = args.work_dir
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    print_freq = args.print_freq
    resume = args.resume if args.resume != '' else None
    warmup_ratio = args.warmup_ratio
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(f'{work_dir}/checkpoints_new', exist_ok=True)
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
    
    writer = SummaryWriter(args.resume)
    
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = MS(args)
    model = model.to(device)
    
    
    
    if resume and os.path.isfile(resume):
        checkpoint = torch.load(resume, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # best_acc = checkpoint['best_acc']
    else:
        logger.warning(f'Không tìm thấy checkpoints tại {resume}')
        start_epoch = 0
        # best_acc = 0
    
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    epochs = args.epochs
    warmup_ratio = args.warmup_ratio
    opt = AdamW(model.parameters(), lr=args.lr)
    batch_size = args.batch_size
    total_steps = (len(train_dataset) // batch_size + 1 ) * epochs
    
    scheduler = get_linear_schedule_with_warmup(opt, 
                                                num_warmup_steps=warmup_ratio * total_steps,
                                            num_training_steps=total_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    

    for epoch in tqdm.tqdm(range(start_epoch, epochs), desc = 'Epochs'):
        model.train()
        #train_dataloader: 1 dict có các key: 'frames', 'vid', 'labels'
        progress_bar = tqdm.tqdm(train_dataloader)
        for step, batch in enumerate(progress_bar):
            labels = batch['labels'].to(device)
            frames = batch['frames'].to(device)
            #labels: tensor[0,1]
            #frames: tensor
            #id: tên của video
            opt.zero_grad()
            #Forward voi mixed precision
            if args.fp16:
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    logits = model(frames)
                    scores = logits.sigmoid()
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                    acc = (scores.round() == labels).float().mean().item()
                    pn = (labels == 1).sum()/len(labels)
                    ap_ = average_precision_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
                    loss, ap, acc, pn = train_step(batch)
                    progress_bar.set_description(f'Epoch {epoch+1}/{epochs}, Step {step+1}/{len(train_dataloader)}, Loss {loss.item()}, Acc {acc.item()}, PN {pn.item()}, AP {ap_}')
                    #Backward:
                    scaler.scale(loss).backward()
                    scaler.step(opt) 
                    scaler.update()
            else:
                logits = model(frames)
                scores = logits.sigmoid()
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                acc = (scores.round() == labels).float().mean().item()
                pn = (labels == 1).sum()/len(labels)
                ap_ = average_precision_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
                
                loss.backward()
                opt.step()
                
            scheduler.step()
            # if step % args.print_freq == 0:
                # print(f'Epoch {epoch}, Step {step}, Loss {loss.item()}, Acc {acc.item()}, PN {pn.item()}, AP {ap_}')
            # if step % 10 == 0:
            logger.info(f'Epoch {epoch + 1}/{epochs}, Step {step + 1}/{len(train_dataloader)}, Loss {loss.item()}, Acc {acc}, PN {pn}, AP {ap_}')
        
        # if epoch % args.eval_freq == 0:
        model.eval()
        tot_val_labels = []
        tot_val_scores = []
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for val_batch in val_dataloader:
                val_labels = val_batch['labels'].to(device)
                val_frames = val_batch['frames'].to(device)
                val_logits = model(val_frames).to(device)
                val_scores = val_logits.sigmoid()
                
                
                tot_val_labels.extend(val_labels.detach().cpu().numpy())
                tot_val_scores.extend(val_scores.detach().cpu().numpy())
                
        # tot_val_labels = np.concatenate(tot_val_labels, axis=0)
        # tot_val_scores = np.concatenate(tot_val_scores, axis=0)
        # In ra kích thước của tot_val_labels và tot_val_scores để kiểm tra
        # Chuyển đổi tot_val_labels và tot_val_scores thành tensor
        tot_val_labels = torch.tensor(tot_val_labels, dtype=torch.float32).to(device)
        tot_val_scores = torch.tensor(tot_val_scores, dtype=torch.float32).to(device)
        
        # # In ra kích thước của tot_val_labels và tot_val_scores để kiểm tra
        # logger.info(f"tot_val_labels shape: {tot_val_labels.shape}")
        # logger.info(f"tot_val_scores shape: {tot_val_scores.shape}")
        
        val_ap = average_precision_score(tot_val_labels.cpu().numpy(), tot_val_scores.cpu().numpy())
        val_acc = (tot_val_scores.round() == tot_val_labels).float().mean().item()
        val_loss = F.binary_cross_entropy(tot_val_scores, tot_val_labels).item()
        
        # print(f'Epoch {epoch}, Val Acc {val_acc}, Val AP {val_ap}')
        logger.info('*******'f'Epoch {epoch + 1}, Val loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AP: {val_ap:.4f}')
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            # 'best_acc': best_acc
        }
        torch.save(checkpoint, f'{args.work_dir}/checkpoints_new/epoch_{epoch + 1 }.pt')
    
    # for epoch in tqdm.tqdm(range(start_epoch, epochs), desc='Epochs'):
    #     model.train()
    #     total_loss = 0
    #     total_acc = 0
    #     total_ap = 0
    #     steps = 0
    
    #     progress_bar = tqdm.tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
    #     for step, batch in enumerate(progress_bar):
    #         labels = batch['labels'].to(device)
    #         frames = batch['frames'].to(device)
            
    #         opt.zero_grad()
            
    #         if args.fp16:
    #             with torch.cuda.amp.autocast(enabled=args.fp16):
    #                 logits = model(frames)
    #                 scores = logits.sigmoid()
    #                 loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                    
    #             scaler.scale(loss).backward()
    #             scaler.unscale_(opt)
    #             clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             scaler.step(opt)
    #             scaler.update()
    #         else:
    #             logits = model(frames)
    #             scores = logits.sigmoid()
    #             loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                
    #             loss.backward()
    #             clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             opt.step()
            
    #         scheduler.step()
            
    #         with torch.no_grad():
    #             acc = (scores.round() == labels).float().mean()
    #             pn = (labels == 1).float().mean()
    #             ap_ = average_precision_score(labels.cpu().numpy(), scores.cpu().numpy())
            
    #         total_loss += loss.item()
    #         total_acc += acc.item()
    #         total_ap += ap_
    #         steps += 1
            
    #         progress_bar.set_postfix({
    #             'Loss': f'{loss.item():.4f}',
    #             'Acc': f'{acc.item():.4f}',
    #             'AP': f'{ap_:.4f}'
    #         })
            
    #         if step % args.print_freq == 0:
    #             logger.info(f'Epoch {epoch + 1}/{epochs}, Step {step + 1}/{len(train_dataloader)}, Loss {loss.item()}, Acc {acc.item()}, PN {pn.item()}, AP {ap_}')
    #     avg_loss = total_loss / steps
    #     avg_acc = total_acc / steps
    #     avg_ap = total_ap / steps
    #     logger.info(f'Epoch {epoch + 1 } - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}, Avg AP: {avg_ap:.4f}')
        
    #     # Validation
    #     model.eval()
    #     val_loss = 0
    #     val_acc = 0
    #     val_ap = 0
    #     val_steps = 0
        
    #     with torch.no_grad():
    #         for val_batch in tqdm.tqdm(val_dataloader, desc='Validation'):
    #             val_labels = val_batch['labels'].to(device)
    #             val_frames = val_batch['frames'].to(device)
                
    #             val_logits = model(val_frames)
    #             val_scores = val_logits.sigmoid()
                
    #             val_loss += F.binary_cross_entropy_with_logits(val_logits, val_labels.float()).item()
    #             val_acc += (val_scores.round() == val_labels).float().mean().item()
    #             val_ap += average_precision_score(val_labels.cpu().numpy(), val_scores.cpu().numpy())
    #             val_steps += 1
        
    #     avg_val_loss = val_loss / val_steps
    #     avg_val_acc = val_acc / val_steps
    #     avg_val_ap = val_ap / val_steps
        
    #     logger.info(f'Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val AP: {avg_val_ap:.4f}')
        
    #     checkpoint = {
    #         'epoch': epoch + 1,
    #         'model': model.state_dict(),
    #         'opt': opt.state_dict(),
    #         'scheduler': scheduler.state_dict(),
    #     }
    #     torch.save(checkpoint, f'{args.work_dir}/checkpoints/epoch_{epoch + 1}.pt')
    
if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    main(args, cfg)
            
                
                
         



    
    
    
        
        
        
        