# Nhập các thư viện cần thiết
import torch
import video.clip as clip
from video.model import MS
from mmcv import Config

# Tệp này có tác dụng chuyển đổi các mô hình PyTorch sang định dạng TorchScript
# để sử dụng trong môi trường sản xuất

if __name__ == '__main__':
    # Đường dẫn đến checkpoint của mô hình CLIP
    checkpoint = r"D:\VideoMatching\checkpoints\clip_vit-l-14"
    # Kích thước ảnh đầu vào
    img_width = 224
    # Tải mô hình CLIP từ checkpoint
    model = clip.from_pretrained(checkpoint)
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    # Chuyển mô hình lên GPU
    model.cuda()
    # Tạo dữ liệu đầu vào giả lập
    dummy_input = torch.randn(1, 3, img_width, img_width).cuda()
    # Chuyển đổi mô hình sang TorchScript
    traced_script_module = torch.jit.trace(model,example_inputs=dummy_input)
    # Lưu mô hình TorchScript
    torch.jit.save(traced_script_module,r"D:\VideoMatching\checkpointss\clip.torchscript.pt")

    # Đường dẫn đến checkpoint của mô hình MS
    checkpoint = r"D:\VideoMatching\train\train_vid_score\checkpoints\epoch_8.pt"
    # Đường dẫn đến file cấu hình
    cfg = r"D:\VideoMatching\train\train_vid_score\config_for_video_score_train.py"
    # Đọc file cấu hình
    cfg = Config.fromfile(cfg)
    # Khởi tạo mô hình MS
    model = MS(cfg)
    # Tải trọng số của mô hình
    state_dict = torch.load(checkpoint,map_location='cpu')['model']
    # Xử lý trọng số để phù hợp với cấu trúc mô hình
    state_dict_ = dict()
    for k,v in state_dict.items():
        if(k.startswith('module.')):
            state_dict_[k[len('module.'):]] = v
        else:
            state_dict_[k] = v
    # Nạp trọng số vào mô hình
    model.load_state_dict(state_dict_)
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    # Chuyển mô hình lên GPU
    model.cuda()
    # Tạo dữ liệu đầu vào giả lập
    dummy_input = torch.randn(1, 256, cfg.feat_dim).cuda()
    # Chuyển đổi mô hình sang TorchScript
    traced_script_module = torch.jit.trace(model,example_inputs=dummy_input)
    # Lưu mô hình TorchScript
    torch.jit.save(traced_script_module,r"D:\VideoMatching\checkpointss\vsm.torchscript.pt")