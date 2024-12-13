# File này định nghĩa các lớp Dataset để xử lý dữ liệu cho các tác vụ học máy và thị giác máy tính.

# Tác dụng chính của file:
# 1. Cung cấp các lớp Dataset tùy chỉnh để làm việc với dữ liệu đa dạng (ảnh, video, cặp dữ liệu).
# 2. Hỗ trợ đọc dữ liệu từ nhiều nguồn khác nhau (local, cloud storage).
# 3. Cung cấp các tiện ích để xử lý và biến đổi dữ liệu.

# Giải thích các thành phần chính:

# 1. base64_encode_image: Hàm mã hóa ảnh thành chuỗi base64.

# 2. ItemDataset: 
#    - Lớp cơ sở để làm việc với các mục dữ liệu đơn lẻ.
#    - Hỗ trợ đọc dữ liệu từ nhiều nguồn và loại dữ liệu khác nhau.

# 3. PairDataset:
#    - Lớp để làm việc với các cặp dữ liệu (query-gallery).
#    - Hữu ích cho các tác vụ như tìm kiếm ảnh hoặc video.

# 4. ImageItemDataset:
#    - Lớp con của ItemDataset, chuyên biệt cho dữ liệu ảnh.
#    - Hỗ trợ áp dụng các phép biến đổi lên ảnh.

# 5. VideoFramesDataset:
#    - Lớp để làm việc với các khung hình video.
#    - Hỗ trợ truy cập ngẫu nhiên vào các khung hình của nhiều video.

# 6. Các hàm tiện ích:
#    - inter_search: Tìm kiếm nhị phân trong danh sách khoảng.
#    - Các phương thức static để tạo khóa cho ảnh và đặc trưng.

# File này đóng vai trò quan trọng trong việc chuẩn bị và quản lý dữ liệu cho các mô hình học máy,
# đặc biệt là trong lĩnh vực xử lý ảnh và video.






# Import các thư viện cần thiết
import base64
from typing import Sequence, Callable

from torch.utils.data import Dataset

from .utils import *

# Hàm mã hóa ảnh thành chuỗi base64
def base64_encode_image(data: np.ndarray) -> str:
    ret, data_bytes = cv2.imencode('.png', data)  # Chuyển đổi ảnh thành định dạng PNG
    data_base64 = str(base64.b64encode(data_bytes, b"-_"), "utf-8")  # Mã hóa dữ liệu thành chuỗi base64
    return data_base64

# Lớp Dataset cho các mục dữ liệu đơn lẻ
class ItemDataset(Dataset):
    def __init__(self,
                 data_list: Sequence[Tuple[str, str]],
                 root: str = "",
                 store_type: str = StoreType.LOCAL.type_name,
                 data_type: str = DataType.BYTES.type_name,
                 trans_key_func: Callable = lambda x: x,
                 use_cache: bool = False,
                 **kwargs):
        self.uuids, self.files = zip(*data_list)  # Tách danh sách dữ liệu thành uuids và files
        self.root = root  # Thư mục gốc
        self.trans_key_func = trans_key_func  # Hàm chuyển đổi khóa

        self.reader = build_reader(store_type, data_type, **kwargs)  # Tạo đối tượng đọc dữ liệu

        self.use_cache = use_cache  # Sử dụng bộ nhớ đệm hay không

    # Đọc một mục dữ liệu
    def read_item(self, idx):
        key = self.files[idx]
        path = self.trans_key_func(key)
        path = os.path.join(self.root, path) if self.root else path
        return self.uuids[idx], self.reader.read(path)

    # Lấy một mục dữ liệu
    def __getitem__(self, item) -> Tuple[str, Any]:
        if self.use_cache:
            logger.info(f"{os.getpid()} cache hit")
            return self.cache[item]
        else:
            return self.read_item(item)

    # Trả về số lượng mục dữ liệu
    def __len__(self):
        return len(self.files)

# Lớp Dataset cho các cặp dữ liệu
class PairDataset(Dataset):
    def __init__(self,
                 query_list: Sequence[Tuple[str, str]] = None,
                 gallery_list: Sequence[Tuple[str, str]] = None,
                 pair_list: Sequence[Tuple[str, str]] = None,
                 file_dict: Dict[str, str] = None,
                 root: str = "",
                 store_type: str = StoreType.LOCAL.type_name,
                 data_type: str = DataType.BYTES.type_name,
                 trans_key_func: Callable = lambda x: x,
                 **kwargs):

        self.query_list = query_list  # Danh sách truy vấn
        self.gallery_list = gallery_list  # Danh sách thư viện
        self.pair_list = pair_list  # Danh sách cặp
        self.file_dict = file_dict  # Từ điển file

        self.root = root  # Thư mục gốc
        self.trans_key_func = trans_key_func  # Hàm chuyển đổi khóa
        self.reader = build_reader(store_type, data_type, **kwargs)  # Tạo đối tượng đọc dữ liệu

    # Lấy một cặp dữ liệu
    def __getitem__(self, item) -> Tuple[str, str, Any, Any]:
        if self.pair_list:
            query_id, gallery_id = self.pair_list[item]

            query_file = self.file_dict[query_id]
            gallery_file = self.file_dict[gallery_id]
        else:
            # Lặp qua tích Descartes của query_list và gallery_list theo thứ tự hàng chính
            i, j = item // len(self.gallery_list), item % len(self.gallery_list)
            query_id, query_file = self.query_list[i]
            gallery_id, gallery_file = self.gallery_list[j]

        query_file = self.trans_key_func(query_file)
        gallery_file = self.trans_key_func(gallery_file)

        query_path = os.path.join(self.root, query_file) if self.root else query_file
        gallery_path = os.path.join(self.root, gallery_file) if self.root else gallery_file

        return query_id, gallery_id, self.reader.read(query_path), self.reader.read(gallery_path)

    # Trả về số lượng cặp dữ liệu
    def __len__(self):
        return len(self.pair_list) if self.pair_list else len(self.query_list) * len(self.gallery_list)

# Lớp Dataset cho các mục dữ liệu ảnh
class ImageItemDataset(ItemDataset):
    def __init__(self,
                 data_list: Sequence[Tuple[str, str]],
                 root: str = "",
                 transforms: List[Any] = None,
                 store_type: str = StoreType.LOCAL.type_name,
                 **kwargs):
        super(ImageItemDataset, self).__init__(data_list,
                                               root=root,
                                               store_type=store_type,
                                               data_type=DataType.IMAGE.type_name, **kwargs)
        self.transforms = transforms  # Danh sách các phép biến đổi

    # Lấy một mục dữ liệu ảnh và áp dụng các phép biến đổi
    def __getitem__(self, item):
        key, value = super().__getitem__(item)
        if self.transforms:
            for t in self.transforms:
                value = t(value)
        return key, value

# Hàm tìm kiếm nhị phân trong danh sách khoảng
def inter_search(val: int, interval_list: List[int]):
    low_ind, high_ind = 0, len(interval_list) - 1

    while high_ind - low_ind > 1:
        mid_ind = (low_ind + high_ind) // 2

        if val > interval_list[mid_ind]:
            low_ind = mid_ind
        elif val < interval_list[mid_ind]:
            high_ind = mid_ind
        else:
            return mid_ind
    return low_ind

# Lớp Dataset cho các khung hình video
class VideoFramesDataset(Dataset):
    def __init__(self, video_list: List[Tuple[str, str, int]],
                 id_to_key_fn: Callable,
                 root: str = "",
                 transforms: List[Any] = None,
                 store_type: str = StoreType.LOCAL.type_name,
                 data_type: str = DataType.IMAGE.type_name,
                 **kwargs):
        super(VideoFramesDataset, self).__init__()
        self.root = root  # Thư mục gốc

        self.reader = build_reader(store_type, data_type, **kwargs)  # Tạo đối tượng đọc dữ liệu

        self.id_to_key_fn = id_to_key_fn  # Hàm chuyển đổi id thành khóa
        self.transforms = transforms  # Danh sách các phép biến đổi
        self.video_list = video_list  # Danh sách video
        frame_cnt_list = [0, *[v[-1] for v in video_list]]
        self.offset_list = np.cumsum(frame_cnt_list).tolist()  # Danh sách offset tích lũy

    # Lấy một khung hình video
    def __getitem__(self, item):
        video_idx, frame_idx = self.offset_to_index(item)

        vid, vdir, _ = self.video_list[video_idx]
        path = self.id_to_key_fn(vdir, frame_idx)
        path = os.path.join(self.root, path) if self.root else path
        value = self.reader.read(path)

        if self.transforms:
            for t in self.transforms:
                value = t(value)
        return vid, frame_idx, value

    # Trả về tổng số khung hình
    def __len__(self):
        return self.offset_list[-1]

    # Chuyển đổi offset thành chỉ số video và khung hình
    def offset_to_index(self, offset: int) -> (int, int):
        video_idx = inter_search(offset, self.offset_list)
        # Các khung hình được lưu trữ bắt đầu từ 0
        frame_idx = offset - self.offset_list[video_idx]
        return video_idx, int(frame_idx)

    # Chuyển đổi chỉ số thành chuỗi
    @staticmethod
    def idx2str(idx: int):
        return f"{idx:05d}"

    # Tạo khóa cho ảnh
    @staticmethod
    def build_image_key(vdir: str, frame_idx: int):
        return os.path.join(vdir, VideoFramesDataset.idx2str(frame_idx) + ".jpg")

    # Tạo khóa cho đặc trưng
    @staticmethod
    def build_feature_key(vdir: str, frame_idx: int):
        return os.path.join(vdir, VideoFramesDataset.idx2str(frame_idx) + ".npy")

