# Import các thư viện cần thiết
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# File này định nghĩa các mô hình neural network để xử lý video
# Có 2 lớp chính: MD (Mô hình Descriptor) và MS (Mô hình Scoring)

# Lớp MD: Mô hình Descriptor để tạo ra các đặc trưng mô tả video
class MD(nn.Module):

    def __init__(self, args):
        super(MD, self).__init__()

        # Lớp projection để chuyển đổi kích thước đặc trưng
        self.frame_proj = nn.Sequential(nn.Linear(args.feat_dim, args.bert_dim), nn.LayerNorm(args.bert_dim))

        # Cấu hình và khởi tạo mô hình BERT
        config = AutoConfig.from_pretrained(args.bert_path)
        config.gradient_checkpointing = args.gradient_checkpointing
        self.bert = AutoModel.from_pretrained(args.bert_path, config=config)
        self.max_frames = args.max_frames

        # Lớp projection cuối cùng để tạo ra embedding
        self.output_proj = nn.Linear(args.bert_dim * 2, args.output_dim)

    def forward(self, feats):
        # Chuyển đổi đặc trưng video
        vision_feats = self.frame_proj(feats)  # b, max_frames, h
        # Tạo mask cho các frame có giá trị
        masks = feats.abs().sum(dim=2).gt(0)

        # Thêm các token đặc biệt (CLS và SEP)
        bz, device = vision_feats.size(0), vision_feats.device
        text = torch.tensor([101, 102], dtype=torch.long)[None].to(device)  # 1, 2
        emb = self.bert.get_input_embeddings()
        text_emb = emb(text).expand((bz, -1, -1))  # bz, 2, hidden
        cls_emb, sep_emb = text_emb[:, 0], text_emb[:, 1]

        # Kết hợp embedding của CLS, video frames, và SEP
        inputs_embeds = torch.cat([cls_emb[:, None], vision_feats, sep_emb[:, None]], dim=1)
        masks = torch.cat([torch.ones((bz, 2)).to(device), masks], dim=1)

        # Đưa qua mô hình BERT
        states = self.bert(inputs_embeds=inputs_embeds, attention_mask=masks)[0]
        # Tính trung bình các frame không bị mask
        avg_pool = self._nonzero_avg_pool(states, masks)
        cls_pool = states[:, 0]
        # Kết hợp embedding CLS và trung bình
        cat_pool = torch.cat([cls_pool, avg_pool], dim=1)
        # Tạo embedding cuối cùng
        embeds = self.output_proj(cat_pool)

        return embeds

    @staticmethod
    def _random_mask_frame(frame, prob=0.15):
        # Hàm để ngẫu nhiên mask một số frame (không được sử dụng trong forward)
        mask_prob = torch.empty(frame.shape[0], frame.shape[1]).uniform_(0, 1).to(device=frame.device)
        mask = (mask_prob < prob).to(dtype=torch.long)
        frame = frame * (1 - mask.unsqueeze(2))
        return frame, mask

    @staticmethod
    def _nonzero_avg_pool(hidden, mask):
        # Hàm tính trung bình các frame không bị mask
        mask = mask.to(hidden.dtype)
        hidden = hidden * mask[..., None]
        length = mask.sum(dim=1, keepdim=True)
        avg_pool = hidden.sum(dim=1) / (length + 1e-5)
        return avg_pool


# Lớp MS: Mô hình Scoring để đánh giá video
class MS(nn.Module):

    def __init__(self, args): 
        super(MS, self).__init__() # Kế thừa từ lớp cha nn.Module của PyTorch

        # Lớp projection để chuyển đổi kích thước đặc trưng từ kích thước feat_dim đầu vào sang kích thước bert_dim để phù hợp với BERT
        self.frame_proj = nn.Sequential(nn.Linear(args.feat_dim, args.bert_dim), nn.LayerNorm(args.bert_dim))

        # Cấu hình và khởi tạo mô hình BERT
        config = AutoConfig.from_pretrained(args.bert_path) # Tải cấu hình mô hình BERT từ đường dẫn args.bert_path
        config.gradient_checkpointing = args.gradient_checkpointing # Sử dụng gradient checkpointing nếu được cấu hình
        self.bert = AutoModel.from_pretrained(args.bert_path, config=config) # Tải mô hình BERT từ đường dẫn args.bert_path
        self.max_frames = args.max_frames # Số lượng frame tối đa

        # Lớp projection cuối cùng để tạo ra điểm số
        self.output_proj = nn.Linear(args.bert_dim * 2, 1) # Lớp Linear với đầu vào là bert_dim * 2 và đầu ra là 1

    def forward(self, feats): #Đầu vào là features, đầu ra là logits(điểm số)
        # Chuyển đổi đặc trưng video
        vision_feats = self.frame_proj(feats)  # Chuyển đổi đặc trưng video từ kích thước feat_dim sang kích thước bert_dim
        # Tạo mask cho các frame có giá trị
        masks = feats.abs().sum(dim=2).gt(0) # Tạo mask cho các frame có giá trị, các frames có tổng giá trị tuyệt đối khác 0 sẽ được mask bằng True

        # Thêm các token đặc biệt (CLS và SEP)
        bz, device = vision_feats.size(0), vision_feats.device # Lấy kích thước batch size (= số lượng video trong batch) và thiết bị
        text = torch.tensor([101, 102], dtype=torch.long)[None].to(device) # Tạo tensor text với giá trị [101, 102] và chuyển về thiết bị device
        emb = self.bert.get_input_embeddings() # Lấy embedding đầu vào của các token từ mô hình BERT
        text_emb = emb(text).expand((bz, -1, -1))  # Mở rộng embedding của các token CLS và SEP cho toàn bộ batch.
        cls_emb, sep_emb = text_emb[:, 0], text_emb[:, 1] # Lưu trữ embedding của token CLS và SEP

        '''
        Đoạn mã này có mục đích tạo ra các token đặc biệt (CLS và SEP) để thêm vào chuỗi embedding của các frame video 
        trước khi đưa chúng vào mô hình BERT.
         Đây là cách thường được sử dụng khi làm việc với mô hình BERT để xử lý các chuỗi dữ liệu không phải văn bản, như video.
        '''

        # Kết hợp embedding của CLS, video frames, và SEP
        inputs_embeds = torch.cat([cls_emb[:, None], vision_feats, sep_emb[:, None]], dim=1) # Kết hợp embedding của token CLS, các frame video, và SEP thành một tensor đầu vào cho BERT.
        masks = torch.cat([torch.ones((bz, 2)).to(device), masks], dim=1) # Cập nhật mask bằng cách thêm các vị trí mask cho CLS và SEP (luôn bằng 1).

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

    @staticmethod
    def _random_mask_frame(frame, prob=0.15):
        # Hàm để ngẫu nhiên mask một số frame (không được sử dụng trong forward)
        mask_prob = torch.empty(frame.shape[0], frame.shape[1]).uniform_(0, 1).to(device=frame.device)
        mask = (mask_prob < prob).to(dtype=torch.long)
        frame = frame * (1 - mask.unsqueeze(2))
        return frame, mask
        '''
        Hàm này được sử dụng để ngẫu nhiên mask một số frame trong video với xác suất prob.
        Điều này giúp mô hình học được cách xử lý các frame bị mất thông tin, phục vụ khả năng tổng quát hóa tốt hơn.
        Tương tự như Dropout 
        '''

    @staticmethod
    def _nonzero_avg_pool(hidden, mask):
        # Hàm tính trung bình các frame không bị mask
        mask = mask.to(hidden.dtype)
        hidden = hidden * mask[..., None]
        length = mask.sum(dim=1, keepdim=True)
        avg_pool = hidden.sum(dim=1) / (length + 1e-5)
        return avg_pool

# Giải thích:
# File này định nghĩa hai mô hình neural network để xử lý video:
# 1. MD (Mô hình Descriptor): Tạo ra các đặc trưng mô tả video
# 2. MS (Mô hình Scoring): Đánh giá và cho điểm video
# 
# Cả hai mô hình đều sử dụng kiến trúc BERT để xử lý chuỗi các frame video.
# Các bước chính trong quá trình xử lý bao gồm:
# - Chuyển đổi đặc trưng video
# - Thêm các token đặc biệt (CLS và SEP)
# - Đưa qua mô hình BERT
# - Tính toán embedding hoặc điểm số cuối cùng
# mask:0 là frame bị mask, 1 là frame không bị mask. Mask là ẩn dữ liệu, giúp mô hình học cách xử lý các frame bị mất thông tin. VÍ dụ:
# mask1: là frame ko bị mask: nghĩa là giữ lại hoặc sử dụng dữ liệu đó. CÒn mask0: bỏ qua hoặc ẩn, các phần tử ko đc đưa vào tinhs toán hoặc xử lí 
# Các mô hình này có thể được sử dụng trong các tác vụ như phân loại video,
# tìm kiếm video dựa trên nội dung, hoặc đánh giá chất lượng video.
