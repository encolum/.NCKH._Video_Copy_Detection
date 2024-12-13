# File này triển khai mô hình CLIP (Contrastive Language-Image Pre-training) cho xử lý video
# Nó định nghĩa các lớp và hàm cần thiết để xây dựng và khởi tạo mô hình CLIP

from collections import OrderedDict
#OrderedDict là một loại từ điển đặc biệt đảm bảo rằng các phần tử sẽ được duy trì theo thứ tự thêm vào.
#Được sử dụng để đảm bảo thứ tự cho các cặp key-value trong những trường hợp quan trọng về thứ tự
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
#mmcv là một thư viện dùng trong computer vision. Hàm load_checkpoint được sử dụng để tải các  (checkpoints)
import torch.utils.checkpoint
#Cung cấp chức năng checkpointing cho mô hình học sâu, tức là chỉ lưu lại các tensor cần thiết để tính toán ngược (backward pass)
# . Điều này giúp tiết kiệm bộ nhớ GPU khi huấn luyện các mô hình lớn.
import yaml
#Tác dụng: Thư viện này dùng để đọc và ghi dữ liệu từ các tệp YAML
import os


class LayerNorm(nn.LayerNorm): #Kế thừa từ nn.LayerNorm
#chuẩn hóa đầu ra của các lớp trong mạng neural.
#là một phương pháp chuẩn hóa trong mạng neural (neural networks), 
# thường được áp dụng sau các phép biến đổi tuyến tính hoặc phi tuyến tính 
# để ổn định quá trình huấn luyện và tăng tốc độ hội tụ.
    def forward(self, x: torch.Tensor):
        # Thực hiện chuẩn hóa lớp và trả về kết quả
        ret = super().forward(x)
        return ret


class QuickGELU(nn.Module):
    # Activation QuickGELU

    def forward(self, x: torch.Tensor):
        # QuickGELu activation function
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    # Khối chú ý dư thừa trong mô hình Transformer
    

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        '''
    d_model: Kích thước (chiều) của các vector đầu vào và đầu ra của mô hình (embedding size).
    n_head: Số lượng "đầu" trong cơ chế chú ý đa đầu (multi-head attention), cho phép mô hình có thể chú ý tới nhiều vùng khác nhau của đầu vào.
    attn_mask: Mặt nạ chú ý (attention mask), thường được sử dụng để bỏ qua hoặc che đi các phần của đầu vào khi thực hiện chú ý, ví dụ như bỏ qua các token không cần thiết trong NLP.
    
    '''
        super().__init__()

        # Khởi tạo các lớp trong khối
        self.attn = nn.MultiheadAttention(d_model, n_head) #MultiheadAttention #tham số cấu hình: d_model, n_head. Ví dụ d_model=512, n_head=8
        self.ln_1 = LayerNorm(d_model) #LayerNorm
        self.mlp = nn.Sequential(OrderedDict([ #MLP
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor): #Tham số x là tensor đầu vào: chiều của tensor đầu vào là d_model
        # Áp dụng cơ chế chú ý
        return self.attn(x, x, x, need_weights=False)[0]
    #x,x, x là query, key, value
    '''
    Lớp nn.MultiheadAttention trong PyTorch thực hiện cơ chế chú ý đa đầu. 
    Nó nhận vào ba tham số chính: query (Q), key (K), và value (V). Đây là những vector được sử dụng để tính toán self-attention (chú ý tự hồi quy).
    '''

    def forward(self, x: torch.Tensor):
        # Truyền qua khối chú ý và MLP
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    '''
    Đầu vào x được chuẩn hóa qua LayerNorm (self.ln_1(x)).
    Dữ liệu đã chuẩn hóa đi qua lớp chú ý đa đầu (self.attention()).
    Kết quả từ chú ý được cộng vào x (residual connection).
    Dữ liệu tiếp tục qua LayerNorm thứ hai (self.ln_2(x)), rồi đi qua mạng MLP.
    Kết quả từ MLP cũng được cộng vào x (residual connection).
    Trả về x sau khi đã đi qua khối Residual Attention.
    '''


class Transformer(nn.Module):
    # Mô hình Transformer: bao gồm nhiều ResidualAttentionBlock, gradient checkpointing và layer freezing-> giúp tăng tốc độ huấn luyện và giảm bộ nhớ
    

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, layer_freeze: int = None,
                 gradient_checkpointing: bool = False):
        '''
        width: Kích thước (chiều) của các vector đầu vào và đầu ra của mô hình (embedding size)
        layers: Số lượng khối ResidualAttentionBlock trong mô hình
        heads: Số lượng "đầu" trong cơ chế chú ý đa đầu (multi-head attention)
        attn_mask: Mặt nạ chú ý (attention mask), thường được sử dụng để bỏ qua hoặc che đi các phần của đầu vào khi thực hiện chú ý, ví dụ như bỏ qua các token không cần thiết trong NLP
        layer_freeze: Số lượng lớp cuối cùng được đóng băng (freeze) trong quá trình huấn luyện
        gradient_checkpointing: Sử dụng gradient checkpointing để giảm bộ nhớ GPU khi huấn luyện mô hình
        '''
        super().__init__()
        self.width = width
        self.layers = layers
        self.layer_freeze = layer_freeze if layer_freeze else layers - 1
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor):
        # Hàm tạo forward tùy chỉnh cho gradient checkpointing
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward
        '''
        Hàm create_custom_forward tạo ra một hàm custom_forward để truyền dữ liệu qua một module cụ thể. 
        Hàm này được sử dụng để hỗ trợ gradient checkpointing.
        '''

        # Truyền qua các khối ResidualAttention
        for idx, layer_module in enumerate(self.resblocks): #Duyệt qua từng layer trong resblocks
            if idx < self.layer_freeze: #Nếu chỉ số của lớp < self.layer_freeze -> lớp được đóng băng -> sử dụng torch.no_grad()
                with torch.no_grad():
                    x = layer_module(x)
            else: #Nếu lớp ko bị đóng băng
                if self.gradient_checkpointing: #và sử dụng gradient checkpointing
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), x) #Sử dụng hàm checkpoint để giảm bộ nhớ GPU
                else: #Nếu ko thì truyền dữ liệu qua lớp thông thường
                    x = layer_module(x)

        return x #Return về tensor x sau khi truyền qua tất cả các lớp
    '''
    Đoạn code này thực hiện việc truyền dữ liệu qua các lớp của mô hình, với một số lớp được đóng băng và một số lớp sử dụng gradient checkpointing để tiết kiệm bộ nhớ. 
    Điều này giúp tối ưu hóa quá trình huấn luyện mô hình, đặc biệt là khi làm việc với các mô hình lớn.
    '''

class CLIPModel(nn.Module):
    # Mô hình CLIP chính
    '''
    input_resolution: Độ phân giải đầu vào của ảnh (số pixel của chiều dài và chiều rộng).
    patch_size: Kích thước của mỗi patch mà ảnh được chia ra để làm đầu vào cho mô hình.
    width: Số chiều của vector đặc trưng trong mô hình (cũng là số kênh đầu ra của lớp tích chập conv1) (số channel).
    layers: Số lượng lớp trong Transformer.
    heads: Số lượng "đầu" (heads) trong cơ chế attention của Transformer.
    output_dim: Kích thước của đầu ra cuối cùng.
    pretrained: Đường dẫn đến mô hình đã được huấn luyện sẵn (nếu có).
    layer_freeze: Số lớp Transformer được "đóng băng" (không cho phép cập nhật trọng số) trong quá trình huấn luyện.
    gradient_checkpointing: Cơ chế tiết kiệm bộ nhớ bằng cách lưu checkpoint trong quá trình tính gradient.
    
    '''

    def __init__(
            self,
            input_resolution: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            output_dim: int,
            pretrained: str = None,
            layer_freeze: int = None,
            gradient_checkpointing: bool = False
    ):

        super().__init__()
        self.pretrained = pretrained

        # Khởi tạo các thành phần của mô hình
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        #conv1: Một lớp tích chập xử lý ảnh đầu vào.

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        #Tham số dùng để tạo embedding cho class, learnable
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        #Tham số đại diện cho embedding vị trí, giúp mô hình biết thứ tự vị trí của từng patch trong ảnh.
        self.ln_pre = LayerNorm(width)
        #Lớp LayerNorm để chuẩn hóa dữ liệu trước khi qua Transformer.
        self.transformer = Transformer(width, layers, heads, layer_freeze=layer_freeze,
                                       gradient_checkpointing=gradient_checkpointing)
        #Transformer: Một mạng Transformer với nhiều khối ResidualAttentionBlock, gradient checkpointing và layer freezing. 

        self.ln_post = LayerNorm(width)
        #Lớp LayerNorm để chuẩn hóa dữ liệu sau khi qua Transformer.
        self.proj = None

    def _init_weights(self, m):
        # Khởi tạo trọng số cho các lớp

        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        '''
        _init_weights(m): Đây là phương thức để khởi tạo trọng số cho từng lớp trong mô hình. 
        Các trọng số của lớp Linear được khởi tạo với phân phối chuẩn, 
        còn các tham số bias và trọng số của lớp LayerNorm được khởi tạo lần lượt là 0 và 1.
        '''
    def init_weights(self):
        # Khởi tạo trọng số cho mô hình

        if isinstance(self.pretrained, str):
            msg = f'load model from: {self.pretrained}'
            print(msg)
            load_checkpoint(self, self.pretrained, strict=False, revise_keys=[(r'^visual\.', '')])
        elif self.pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
        '''
        Nếu có mô hình pre-trained, nó sẽ được tải vào  
        nếu không thì khởi tạo trọng số bằng _init_weights.
        '''

    def forward(self, x: torch.Tensor):
        # Truyền dữ liệu qua mô hình *:batch_size
        #shape x: [*,3, height, width]
        x = self.conv1(x)  # shape = [*, width, grid, grid] #grid = height // patch_size, width // patch_size, patch_size là trong conv1
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] ->Hoán vị tensor thành [batch_size, grid * grid, width].
        x = torch.cat(                          #torch.zeroes.shape(batch_size, 1, width -> dtype = x.dtype, device = x.device)
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x],
            dim=1 # -> -> ->Thêm class_embedding vào tensor toàn số 0 để tạo ra một embedding lớp có kích thước [batch_size,1,width]
        )  
        #Kết hợp embedding lớp với tensor đầu vào x: shape = [*, grid ** 2 + 1, width] do dim = 1
        x = x + self.positional_embedding.to(x.dtype)
        '''
        self.positional_embedding là một tensor chứa các embedding vị trí, 
        giúp mô hình học được thông tin về vị trí của các patch trong ảnh.
        Dòng mã này thực hiện việc thêm một embedding vị trí vào tensor đầu vào x
        Mục đích của việc thêm embedding vị trí vào tensor đầu vào là để cung cấp thông tin về vị trí của các patch trong ảnh cho mô hình. 
        Điều này rất quan trọng trong các mô hình transformer, 
        vì chúng không có khả năng tự động học được thông tin về vị trí như các mô hình CNN.
        '''
        x = self.ln_pre(x)
        #Layernorm trước khi đi qua transformer

        x = x.permute(1, 0, 2)  # NLD -> LND
        #Hoán vị chiều của tensor x từ [batch_size, seq_len, width] (NLD) sang dạng [seq_len, batch_size, width] (LND) để phù hợp với đầu vào của transformer.
        '''
        seq_len = grid**2 + 1 
        Trong đoạn mã này, seq_len (sequence length) là chiều dài của chuỗi đầu vào cho mô hình transformer. 
        Đây là số lượng các patch (hoặc token) trong ảnh sau khi đã được chia nhỏ và thêm embedding lớp.
        '''
        x = self.transformer(x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        #hoán vị tensor trở lại dạng ban đầu
        x = self.ln_post(x)
        #Layernorm sau khi đi qua transformer

        if self.proj is not None:
            x = x @ self.proj
        '''
        Nếu self.proj không phải là None, thực hiện phép chiếu tuyến tính bằng cách nhân tensor x với ma trận chiếu [self.proj]
        '''

        return x


def from_pretrained(config_dir):
    # Tạo mô hình CLIP từ cấu hình và trọng số đã được huấn luyện trước
    '''
    Đoạn mã này định nghĩa một hàm from_pretrained để tạo một mô hình CLIP từ cấu hình và trọng số đã được huấn luyện trước.
    '''
    #Input: config_dir: Đường dẫn đến thư mục chứa cấu hình và trọng số của mô hình đã đc huấn luyện trước
    config_path = os.path.join(config_dir, "config.yaml")
    ckpt_path = os.path.join(config_dir, "pytorch_model.bin")
    '''
    config_path: Đường dẫn đến tệp cấu hình config.yaml.
    ckpt_path: Đường dẫn đến tệp trọng số pytorch_model.bin.
    '''
    with open(config_path, 'r', encoding="utf-8") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    #-> Đọc tệp cấu hình
    yaml_cfg["pretrained"] = ckpt_path
    #Thêm đường dẫn đến trọng số vào cấu hình
    model = CLIPModel(**yaml_cfg)
    '''
    Tạo một đối tượng mô hình CLIP bằng cách sử dụng cấu hình đã được đọc và bổ sung. Các tham số trong yaml_cfg được truyền vào hàm khởi tạo của lớp CLIPModel.
    '''
    model.init_weights()
    '''
    Gọi phương thức init_weights của mô hình để khởi tạo trọng số từ tệp trọng số pytorch_model.bin.
    '''

    return model

