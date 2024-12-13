from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import time
import zipfile
from zipfile import ZipFile
import io
import argparse
import io
import os
import time
import zipfile
from zipfile import ZipFile

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vids_path', '-v', type = str, default = r"D:\VideoMatching\data\meta\vids.txt")
    parser.add_argument('--zip_prefix', '-z', type = str, default = r"D:\VideoMatching\data\jpg_zips")
    parser.add_argument('--model_path', '-m', type = str, default =  r"D:\VideoMatching\checkpoints\clip_vit-l-14")
    parser.add_argument('--batch_size', '-b', type = int, default = 2)
    parser.add_argument('--max_video_frames', '-f', type = int, default = 256)
    parser.add_argument('--output_path', '-o', type = str, default = r"D:\VideoMatching\data\feat_zips\feats.zip")
    return parser.parse_args()


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _convert_image_to_rgb(image):
    return image.convert("RGB")
def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC), #Resize kích thước ảnh về n_px, sử dụng BICUBIC
        CenterCrop(n_px), #Cắt ảnh ở giữa với kích thước n_px
        _convert_image_to_rgb, #Chuyển ảnh sang định dạng RGB
        ToTensor(), #Chuyển ảnh sang tensor
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # Chuẩn hoá, các số trên là mean,std của RGB
    ])
# Build dataset 
class Data(Dataset):
    def __init__(self, 
                 vids, 
                 zip_prefix, 
                 transform = None, 
                 processor = None,
                 args = None,
                 ):
        self.vids = vids #Danh sách các video
        self.zip_prefix = zip_prefix #Đường dẫn đến file zip chứa các ảnh
        self.transform = transform
        self.processor = processor
        self.args = args
    def __len__(self):
        return len(self.vids)
    def __getitem__(self, idx):
        vid = self.vids[idx]
        zip_path = '{}/{}/{}.zip'.format(self.zip_prefix, vid[-2:], vid)
        img_tensor = torch.zeros(self.args.max_video_frames, 3,224,224) #Img_tensor đại diện cho 1 video, 256: số frames mà 1 video có thể có. Mỗi frame là 1 tensor 3x224x224
        video_mask = torch.zeros(self.args.max_video_frames).long() #Dùng để đánh dấu các frame hợp lệ của video, 0: frame không hợp lệ, 1: frame hợp lệ. Nếu 1 video có ít hơn 256 frames thì các frame thừa sẽ được đánh dấu là 0
        
        try:
            with ZipFile(zip_path, 'r') as handler:
                img_name_list = handler.namelist()
                img_name_list = sorted(img_name_list)
                
                for i, img_name in enumerate(img_name_list):
                    i_img_content = handler.read(img_name)
                    i_img = Image.open(io.BytesIO(i_img_content))
                    if self.processor:
                        i_img_tensor = self.processor(images=i_img, return_tensors="pt").pixel_values.squeeze()
                    elif self.transform:
                        i_img_tensor = self.transform(i_img)
                    else:
                        i_img_tensor = ToTensor()(i_img)
                    img_tensor[i] = i_img_tensor #Mỗi 1 frame ảnh, tương ứng với 1 tensor 3x224x224 và được lưu vào img_tensor
                    video_mask[i] = 1 #Đánh dấu frame hợp lệ
            return img_tensor, video_mask, vid
        except FileNotFoundError as e:
            print(f'File not found: {zip_path}, error: {e}')
            return torch.zeros(1), torch.zeros(1), "error"
        

def main(args):
    args = parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = CLIPModel.from_pretrained(args.model_path)
    processor = CLIPProcessor.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    # Tắt gradient computation để tiết kiệm bộ nhớ
    torch.set_grad_enabled(False)
    
    with open(args.vids_path, 'r', encoding= 'utf-8') as f:
        vids = [x.strip() for x in f]
    
    dataset = Data(vids, args.zip_prefix, transform= transform(224), args = args)
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        drop_last = False,
        num_workers= 0
    )
        
    output_path = args.output_path
    #Model
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as output_handler:
        vid_set = set()
        for k, batch in tqdm(enumerate(data_loader)):
            img = batch[0].to(device) #img size: [2,256,3,224,224]
            video_mask = batch[1].to(device) #video_mask size: [2,256]
            vids = batch[2] #vid size: 2 ID của 2 video
            
            with torch.no_grad():
                inputs = processor(images = img.view(-1,3,224,224), return_tensors = 'pt', padding = True).to(device) #Chuyển đổi tensor thành kích thước phù hợp với CLIP (512 * 3 * 224 * 224)
                outputs = model.get_image_features(**inputs) #trích xuất đăcj trưng của ảnh từ input -> Shape: [2*256, 512] (batch_size*max_video_frames, feat_dim)
                
                outputs = outputs.view(img.shape[0], img.shape[1], -1) # [2,256,512]
                #video_mask: [2,256] gồm 256 số 0 hoặc 1 thêm unsqueeze(-1) để có shape [2,256,1]
                #outputs: [2,256,512] gồm 2 video, mỗi video có 256 frame, mỗi frame có 512 features
                masked_outputs = outputs * video_mask.unsqueeze(-1) #Masking các frame không hợp lệ 
            for i, vid in enumerate(vids):
                if vid not in vid_set:
                    vid_set.add(vid)
                    embedding = masked_outputs[i].cpu().numpy() #Chuyển tensor của frame hợp lệ về numpy array
                    buffer = io.BytesIO() #Tạo 1 buffer đối tượng BytesIO để lưu embedding dưới dạng nhị phân
                    np.save(buffer, embedding) #Lưu mảng numpy embedding vào buffer dưới dạng nhị phân
                    output_handler.writestr(f'{vid}', buffer.getvalue()) #buffer.getvalue(): lấy giá trị của buffer dưới dạng nhị phân và lưu vào file zip
                    
            torch.cuda.empty_cache()
            print(f'Batch {k}. Total unique videos: {len(vid_set)}')
    print(f'Embeddings are saved at {output_path}')
                    
if __name__ == '__main__':
    args = parse_args()
    main(args)
            
    
        
                    
                    
        
        
        
        