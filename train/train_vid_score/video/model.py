import logging
import sys, os
import argparse
import random
import pandas as pd
import torch
import torch.nn as nn
# from vsc.baseline.model_factory.utils import build_dataset
import torch.nn.functional as F
# from video.model import MS
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score
from mmcv import Config
from transformers import AutoTokenizer, AutoModel, AutoConfig

class MS(nn.Module):
    def __init__(self,args):
        super().__init__()
        # Lớp projection để chuyển đổi kích thước đặc trưng từ feat_dim đầu vào sang kích thước bert_dim để phù hợp với BERT rồi layer norm
        self.frame_proj = nn.Sequential(
            nn.Linear(args.feat_dim, args.bert_dim),
            nn.LayerNorm(args.bert_dim)
        )
        # Cấu hình và khởi tạo mô hình BERT
        config = AutoConfig.from_pretrained(args.bert_path)
        # self.gradient_checkpointing = args.gradient_checkpointing
        self.bert = AutoModel.from_pretrained(args.bert_path, config=config)
        self.max_frames = args.max_frames
        
        # Lớp projection cuối cùng để tạo ra điểm số
        self.output_proj = nn.Linear(args.bert_dim * 2, 1)
        
    def forward(self,feats): #Đầu vào là features ở đây là các embedding, đầu ra là logits(điểm số)
        #Shape feats = [batch_size, max_frames, feat_dim] = [2,256,512] la embedding output trong my_extract_feat.py
        # Chuyển đổi đặc trưng video
        #Shape vision_feats = [batch_size, max_frames, bert_dim] = [2,256,768]
        print('Shape feats:',feats.shape)
        vision_feats = self.frame_proj(feats) # Chuyển đổi đặc trưng video từ kích thước feat_dim sang kích thước bert_dim
        # Tạo mask cho các frame có giá trị
        print('Shape vision_feats:',vision_feats.shape)
        masks = feats.abs().sum(dim=2).gt(0)  #Lấy giá trị tuyệt đối của feats, cộng theo chiều thứ 2( giá trị của mỗi frame), nếu > 0 thì True -> hợp lệ
        #CLS: embedding của cả video, SEP: embedding của mỗi frame
        #CLS: Dùng khi cần biểu diễn toàn bộ video, SEP: Dùng khi cần biểu diễn từng frame (ý là cần phân chia các phần(frames) khác nhau trong 1 video)

        #Thêm các token đặc biệt (CLS và SEP)
        bz, device = vision_feats.size(0), vision_feats.device #
        text = torch.tensor([101, 102], dtype=torch.long)[None].to(device) # Tạo tensor text với giá trị [101, 102] và chuyển về thiết bị device
        '''
        Đây là 2 token đặc biệt trong BERT:
        Token [CLS] (101) được thêm vào để biểu diễn toàn bộ đoạn đặc trưng video (feats), giúp mô hình hiểu rằng đây là điểm bắt đầu của chuỗi đặc trưng.
        Token [SEP] (102) được thêm vào để đánh dấu kết thúc chuỗi đặc trưng video.
        '''
        emb = self.bert.get_input_embeddings() # Lấy embedding đầu vào của các token từ mô hình BERT
        text_emb = emb(text).expand((bz, -1, -1)) # Mở rộng embedding của các token CLS và SEP cho toàn bộ batch.
        cls_emb, sep_emb = text_emb[:, 0], text_emb[:, 1]
        
        #Kết hợp embedding của CLS, video frames, và SEP
        #Shape inputs_embeds = [batch_size, max_frames+2, bert_dim] = [2,258,768]
        inputs_embeds = torch.cat([cls_emb[:, None], vision_feats, sep_emb[:, None]], dim=1)
        #Shape masks = [batch_size, max_frames+2] = [2,258]
        masks = torch.cat([torch.ones((bz, 2)).to(device), masks], dim=1)
        
        # Đưa qua mô hình BERT
        states = self.bert(inputs_embeds=inputs_embeds, attention_mask=masks)[0] # Đưa chuỗi embedding  qua mô hình BERT để tạo ra các hidden states
        # Tính trung bình các frame không bị mask
        avg_pool = self._nonzero_avg_pool(states, masks)
        cls_pool = states[:, 0] # Lấy embedding của token CLS
        # Kết hợp embedding CLS và trung bình
        cat_pool = torch.cat([cls_pool, avg_pool], dim=1)
        # Tạo điểm số cuối cùng
        logits = self.output_proj(cat_pool).squeeze(1)

        return logits
    
    def _nonzero_avg_pool(self, hidden, mask):
        # Hàm tính trung bình các frame không bị mask
        mask = mask.to(hidden.dtype)
        hidden = hidden * mask[..., None]
        length = mask.sum(dim=1, keepdim=True)
        avg_pool = hidden.sum(dim=1) / (length + 1e-5)
        return avg_pool