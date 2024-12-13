# Import các thư viện cần thiết
import os
import io
import json
import oss2
import cv2
import configparser
import enum
import numpy as np

from PIL import Image
from typing import Any, Union, Dict, List, Tuple
from multiprocessing import Queue, Process, Pool
from loguru import logger

# File này định nghĩa các lớp và hàm để đọc và ghi dữ liệu từ nhiều nguồn khác nhau như local file system và OSS (Object Storage Service).
# Nó cung cấp các Reader và Writer cho các loại dữ liệu khác nhau như bytes, ảnh, numpy array và JSON.
# Ngoài ra, nó còn có một lớp AsyncWriter để xử lý ghi dữ liệu bất đồng bộ.



#OSS là một dịch vụ lưu trữ đám mây cho phép lưu trữ và truy xuất dữ liệu thông qua internet. 
# Nó thường được sử dụng để lưu trữ các tệp lớn như hình ảnh, video, hoặc bất kỳ loại dữ liệu nào.

#Bucket là một container trong OSS để lưu trữ các đối tượng (objects). 
# Mỗi bucket có một tên duy nhất và có thể chứa nhiều đối tượng.

#StoreType là một enum định nghĩa các loại lưu trữ khác nhau.

#DataType là một enum định nghĩa các loại dữ liệu khác nhau mà hệ thống có thể xử lý.

#Enum (viết tắt của Enumeration) là một kiểu dữ liệu đặc biệt trong lập trình, 
# được sử dụng để định nghĩa một tập hợp các hằng số có tên.


# Định nghĩa enum StoreType để xác định loại lưu trữ
class StoreType(enum.Enum):

    def __new__(cls, *args, **kwargs):
        # Tạo giá trị mới cho enum
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, type_name: str):
        # Khởi tạo enum với tên loại
        self.type_name = type_name

    # Các loại lưu trữ
    LOCAL = "local"  # Lưu trữ cục bộ
    OSS = "oss"  # Lưu trữ trên Object Storage Service


# Định nghĩa enum DataType để xác định loại dữ liệu
class DataType(enum.Enum):

    def __new__(cls, *args, **kwargs):
        # Tạo giá trị mới cho enum
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, type_name: str):
        # Khởi tạo enum với tên loại
        self.type_name = type_name

    # Các loại dữ liệu
    BYTES = "bytes"  # Dữ liệu dạng bytes
    IMAGE = "image"  # Dữ liệu ảnh
    NUMPY = "numpy"  # Dữ liệu numpy array
    JSON = "json"  # Dữ liệu JSON
    DUMMY = "dummy"  # Dữ liệu giả


# Đọc cấu hình OSS từ file
def read_oss_config(path: str) -> Dict[str, Any]:
    # Đọc file cấu hình OSS và trả về thông tin xác thực
    oss_src_config = configparser.ConfigParser()
    oss_src_config.read(os.path.expanduser(path))
    return oss_src_config['Credentials']


# Tạo bucket OSS từ cấu hình
def create_oss_bucket(oss_config: Union[Dict[str, str], str]) -> oss2.Bucket:
    # Tạo và trả về một bucket OSS dựa trên cấu hình
    if isinstance(oss_config, str):
        oss_config = read_oss_config(oss_config)

    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])

    return oss2.Bucket(auth, endpoint=oss_config['endpoint'], bucket_name=oss_config['bucket'])


# Lớp cơ sở cho các Reader
class Reader(object):
    def read(self, path):
        # Phương thức đọc cần được triển khai bởi các lớp con
        raise NotImplementedError


# Đọc dữ liệu từ OSS
def oss_read(bucket, key, oss_root=None):
    # Đọc dữ liệu từ OSS bucket
    path = os.path.join(oss_root, key) if oss_root else key
    data_bytes = bucket.get_object(path).read()
    return data_bytes


# Reader cho OSS
class OssReader(Reader):
    def __init__(self, oss_config: str):
        # Khởi tạo OssReader với cấu hình OSS
        self.bucket = create_oss_bucket(oss_config)

    def read(self, path):
        # Đọc dữ liệu từ OSS
        return oss_read(self.bucket, path)


# Reader cho local file system
class LocalReader(Reader):
    def read(self, path):
        # Đọc dữ liệu từ file cục bộ
        return open(path, 'rb').read()


# Reader giả lập, chỉ trả về đường dẫn
class DummyReader(Reader):
    def __init__(self, reader: Reader):
        # Khởi tạo DummyReader với một Reader khác
        self.reader = reader

    def read(self, path):
        # Trả về đường dẫn mà không đọc dữ liệu
        return path


# Reader cho dữ liệu dạng bytes
class BytesReader(Reader):
    def __init__(self, reader: Reader):
        # Khởi tạo BytesReader với một Reader khác
        self.reader = reader

    def read(self, path):
        # Đọc dữ liệu dạng bytes
        return self.reader.read(path)


# Reader cho dữ liệu dạng ảnh
class ImageReader(Reader):
    def __init__(self, reader: Reader):
        # Khởi tạo ImageReader với một Reader khác
        self.reader = reader

    def read(self, path) -> np.ndarray:
        # Đọc dữ liệu ảnh và trả về dưới dạng numpy array
        return Image.open(io.BytesIO(self.reader.read(path)))


# Reader cho dữ liệu dạng numpy
class NumpyReader(Reader):
    def __init__(self, reader: Reader):
        # Khởi tạo NumpyReader với một Reader khác
        self.reader = reader

    def read(self, path) -> Union[np.ndarray, dict]:
        # Đọc dữ liệu numpy từ file .npy hoặc .npz
        if path.endswith("npz"):
            with np.load(io.BytesIO(self.reader.read(path))) as data:
                return dict(data)
        return np.load(io.BytesIO(self.reader.read(path)))


# Reader cho dữ liệu dạng JSON
class JsonReader(Reader):
    def __init__(self, reader: Reader):
        # Khởi tạo JsonReader với một Reader khác
        self.reader = reader

    def read(self, path) -> Union[np.ndarray, dict]:
        # Đọc dữ liệu JSON
        return json.load(io.BytesIO(self.reader.read(path)))


# Hàm tạo Reader phù hợp dựa trên loại lưu trữ và loại dữ liệu
def build_reader(store_type: str, data_type: str, **kwargs) -> Reader:
    # Tạo Reader phù hợp dựa trên loại lưu trữ và loại dữ liệu
    if store_type == StoreType.LOCAL.type_name:
        reader = LocalReader()
    elif store_type == StoreType.OSS.type_name:
        reader = OssReader(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")

    if data_type == DataType.BYTES.type_name:
        return BytesReader(reader)
    elif data_type == DataType.IMAGE.type_name:
        return ImageReader(reader)
    elif data_type == DataType.NUMPY.type_name:
        return NumpyReader(reader)
    elif data_type == DataType.JSON.type_name:
        return JsonReader(reader)
    elif data_type == DataType.DUMMY.type_name:
        return DummyReader(reader)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


# Lớp cơ sở cho các Writer
class Writer(object):
    def write(self, path: str, data: Any):
        # Phương thức ghi cần được triển khai bởi các lớp con
        raise NotImplementedError


# Writer cho OSS
class OssWriter(Writer):
    def __init__(self, oss_config: str):
        # Khởi tạo OssWriter với cấu hình OSS
        self.bucket = create_oss_bucket(oss_config)

    def write(self, path, data: bytes):
        # Ghi dữ liệu lên OSS
        return self.bucket.put_object(path, data)


# Writer cho local file system
class LocalWriter(Writer):
    def write(self, path, obj: bytes):
        # Ghi dữ liệu vào file cục bộ
        return open(path, 'wb').write(obj)


# Writer cho dữ liệu dạng bytes
class BytesWriter(Writer):
    def __init__(self, writer: Writer):
        # Khởi tạo BytesWriter với một Writer khác
        self.writer = writer

    def write(self, path: str, data: Union[bytes, str]):
        # Ghi dữ liệu dạng bytes
        if isinstance(data, str):
            data = data.encode('utf-8')
        return self.writer.write(path, data)


# Writer cho dữ liệu dạng ảnh
class ImageWriter(Writer):
    def __init__(self, writer: Writer):
        # Khởi tạo ImageWriter với một Writer khác
        self.writer = writer

    def write(self, path: str, data: np.ndarray):
        # Ghi dữ liệu ảnh
        ext = os.path.splitext(path)[-1]
        ret, img = cv2.imencode(ext, data)
        return self.writer.write(path, img.tobytes())


# Writer cho dữ liệu dạng numpy
class NumpyWriter(Writer):
    def __init__(self, writer: Writer):
        # Khởi tạo NumpyWriter với một Writer khác
        self.writer = writer

    def write(self, path:str, data: Union[np.ndarray, dict]):
        # Ghi dữ liệu numpy
        output = io.BytesIO()

        if path.endswith("npz"):
            if isinstance(data, list):
                np.savez(output, *data)
            elif isinstance(data, dict):
                np.savez(output, **data)
            else:
                raise ValueError('invalid type: {} to save to {}', type(data), path)
        else:
            if isinstance(data, np.ndarray):
                np.save(output, data)
            else:
                raise ValueError('invalid type: {} to save to {}', type(data), path)
        output = output.getvalue()

        return self.writer.write(path, output)


# Writer cho dữ liệu dạng JSON
class JsonWriter(Writer):
    def __init__(self, writer: Writer):
        # Khởi tạo JsonWriter với một Writer khác
        self.writer = writer

    def write(self, path: str, data: Union[List, Dict, bytes]):
        # Ghi dữ liệu JSON
        if isinstance(data, list) or isinstance(data, dict):
            output = json.dumps(data, ensure_ascii=False).encode(encoding='utf-8')
        elif isinstance(data, bytes):
            output = data
        elif isinstance(data, str):
            output = data.encode('utf-8')
        else:
            raise ValueError('invalid type: {} to save to {}', type(data), path)

        return self.writer.write(path, output)


# Hàm tạo Writer phù hợp dựa trên loại lưu trữ và loại dữ liệu
def build_writer(store_type: str, data_type: str, **kwargs) -> Writer:
    # Tạo Writer phù hợp dựa trên loại lưu trữ và loại dữ liệu
    if store_type == StoreType.LOCAL.type_name:
        writer = LocalWriter()
    elif store_type == StoreType.OSS.type_name:
        writer = OssWriter(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")

    if data_type == DataType.BYTES.type_name:
        return BytesWriter(writer)
    elif data_type == DataType.IMAGE.type_name:
        return ImageWriter(writer)
    elif data_type == DataType.NUMPY.type_name:
        return NumpyWriter(writer)
    elif data_type == DataType.JSON.type_name:
        return JsonWriter(writer)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


# Lớp Writer bất đồng bộ
class AsyncWriter(object):
    def __init__(self, pool_size: int, store_type: str, data_type: str, **config):
        # Khởi tạo AsyncWriter với số lượng worker, loại lưu trữ và loại dữ liệu
        self.pool_size = pool_size
        self.writer = build_writer(store_type=store_type, data_type=data_type, **config)

        self.in_queue = Queue()
        self.eof_sig = [None, None]

        def worker_loop(writer: Writer, in_queue: Queue):
            # Vòng lặp xử lý của worker
            while True:
                path, data = in_queue.get()

                if path is None and data is None:
                    logger.info("Finish processing, exit...")
                    break

                writer.write(path, data)

        self.workers = []
        for _ in range(self.pool_size):
            # Tạo và khởi động các worker process
            p = Process(target=worker_loop, args=(self.writer, self.in_queue))
            p.start()
            self.workers.append(p)

    def consume(self, data: Tuple[str, Any]):
        # Thêm dữ liệu vào hàng đợi để xử lý
        self.in_queue.put(data)

    def stop(self):
        # Dừng tất cả các worker process
        for _ in range(self.pool_size):
            self.in_queue.put(self.eof_sig)

        for p in self.workers:
            p.join()

