import argparse
import io
import os
import time
import zipfile
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

import clip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vids_path', '-v', type=str, default=r"D:\VideoMatching\data\meta\vids.txt")
    parser.add_argument('--zip_prefix', '-z', type=str, default=r"D:\VideoMatching\data\jpg_zips")
    parser.add_argument('--model_path', '-m', type=str, default=r"D:\VideoMatching\checkpoints\clip_vit-l-14")
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--max_video_frames', '-f', type=int, default=256)
    parser.add_argument('--output_path', '-o', type=str, default=r"D:\VideoMatching\data\feat_zips\feats.zip")
    return parser.parse_args()

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class D(torch.utils.data.Dataset):
    def __init__(self, vids, zip_prefix, preprocess=None, args=None):
        self.transform = preprocess
        self.zip_prefix = zip_prefix
        self.vids = vids
        self.args = args

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        vid = self.vids[index]
        zip_path = "%s/%s/%s.zip" % (self.zip_prefix, vid[-2:], vid)
        img_tensor = torch.zeros(self.args.max_video_frames, 3, 224, 224)
        video_mask = torch.zeros(self.args.max_video_frames).long()

        try:
            with ZipFile(zip_path, 'r') as handler:
                img_name_list = handler.namelist()
                img_name_list = sorted(img_name_list)

                for i, img_name in enumerate(img_name_list):
                    i_img_content = handler.read(img_name)
                    i_img = Image.open(io.BytesIO(i_img_content))
                    i_img_tensor = self.transform(i_img)
                    img_tensor[i, ...] = i_img_tensor
                    video_mask[i] = 1
        except FileNotFoundError as e:
            print(e)

        return img_tensor, video_mask, vid

def main(args):
    s_time = time.time()
    
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải mô hình CLIP
    model = clip.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # Đọc danh sách video
    with open(args.vids_path, "r", encoding="utf-8") as f:
        vids = [x.strip() for x in f]

    # Tạo dataset và dataloader
    infer_dataset = D(vids, args.zip_prefix, preprocess=_transform(224), args=args)
    infer_dataloader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=False,
    )

    # Mở file output để ghi kết quả
    output_handler = zipfile.ZipFile(args.output_path, 'w', compression=zipfile.ZIP_STORED)

    vid_set = set()
    # Lặp qua từng batch dữ liệu
    for k, batch in tqdm(enumerate(infer_dataloader)):
        img = batch[0].to(device)
        video_mask = batch[1].to(device)
        vids = batch[2]

        # Trích xuất đặc trưng sử dụng mô hình CLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            frame_num = video_mask.sum(dim=1).long()
            flat_frames = img[video_mask.bool()]
            flat_feature = model(flat_frames)
            flat_feature = flat_feature[:, 0]

            tot = 0
            stack_feature = []
            for n in frame_num:
                n = int(n)
                real_feat = flat_feature[tot: tot + n]
                feat = F.pad(real_feat, pad=(0, 0, 0, args.max_video_frames - real_feat.size(0)))
                tot += n
                stack_feature.append(feat)
            out_feature = torch.stack(stack_feature, dim=0)
            out_feature = out_feature * video_mask[..., None]
            out_feature = out_feature.reshape(-1, args.max_video_frames, out_feature.size(-1))

        # Chuyển đổi và lưu features
        features = out_feature.cpu().detach().numpy().astype(np.float16)
        assert features.shape[0] == len(vids)

        # Lưu kết quả vào file zip
        for i in range(features.shape[0]):
            vid = vids[i]
            if vid in vid_set:
                continue
            vid_set.add(vid)
            ioproxy = io.BytesIO()
            np.save(ioproxy, features[i])
            npy_str = ioproxy.getvalue()
            output_handler.writestr(vid, npy_str)
            
        torch.cuda.empty_cache()
        print(f'Batch {k}. Total unique videos: {len(vid_set)}')

    output_handler.close()
    print(f"Embeddings saved at {args.output_path}")
    print(f"Total time: {time.time() - s_time:.2f} seconds")

if __name__ == "__main__":
    args = parse_args()
    main(args)