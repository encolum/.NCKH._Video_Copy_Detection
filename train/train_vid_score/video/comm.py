"""
File này chứa các hàm nguyên thủy để giao tiếp đa GPU.
Nó hữu ích khi thực hiện huấn luyện phân tán.
"""

import pickle
'''
pickle: là một module trong Python dùng để serialize và deserialize các đối tượng Python thành các byte strings và ngược lại.
'''

import torch
import torch.distributed as dist
'''
torch.distributed: là một module trong PyTorch dùng để huấn luyện mô hình mạng nơ-ron trên nhiều GPU hoặc máy tính.
'''

def get_world_size():
    if not dist.is_available(): #Check xem module torch.distributed có sẵn không? Nếu không(ko hỗ trợ phân tán) -> retun 1 -> chế độ đơn tiến trình
        return 1
    if not dist.is_initialized(): #Check xem module torch.distributed đã được khởi tạo chưa? Nếu chưa -> return 1 -> chế độ đơn tiến trình
        return 1
    return dist.get_world_size() #Trả về số tiến trình trong nhóm phân tán (bao nhiêu GPU đang sử dụng)


def get_rank():
    # Trả về thứ hạng của tiến trình hiện tại trong nhóm phân tán
    '''
    Thứ hạng (rank) là số thứ tự của một tiến trình trong toàn bộ nhóm phân tán. 
    Mỗi tiến trình trong nhóm được gán một giá trị "rank" duy nhất.
    0: main process
    '''

    if not dist.is_available(): #Check xem module torch.distributed có sẵn không? Nếu không(ko hỗ trợ phân tán) -> retun 0 -> main process
        return 0
    if not dist.is_initialized(): #Check xem module torch.distributed đã được khởi tạo chưa? Nếu chưa -> return 0 -> main process
        return 0
    return dist.get_rank() #Trả về thứ hạng của tiến trình hiện tại trong nhóm phân tán


def is_main_process():
    # Kiểm tra xem tiến trình hiện tại có phải là tiến trình chính (rank 0) hay không
    return get_rank() == 0


def synchronize():
    """
    Hàm trợ giúp để đồng bộ hóa (rào cản) giữa tất cả các tiến trình
    khi sử dụng huấn luyện phân tán
    """
    if not dist.is_available(): 
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data, device="cuda"):
    """
    Thực hiện all_gather trên dữ liệu có thể pickle tùy ý (không nhất thiết phải là tensors)
    Args:
        data: bất kỳ đối tượng có thể pickle nào
    Returns:
        list[data]: danh sách dữ liệu được thu thập từ mỗi rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Serialize thành Tensor
    buffer = pickle.dumps(data) #Chuyển data về 1 chuỗi bytes
    storage = torch.ByteStorage.from_buffer(buffer) #Chuyển chuỗi bytes thành ByteStorage trong pytorch
    tensor = torch.ByteTensor(storage).to(device) #Chuyển ByteStorage thành Tensor

    # Lấy kích thước Tensor của mỗi rank
    local_size = torch.LongTensor([tensor.numel()]).to(device) #là số phần tử trong tensor của tiến trình hiện tại.
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)] # là danh sách chứa kích thước của tensor từ tất cả các tiến trình.
    dist.all_gather(size_list, local_size) #thu thập kích thước từ mọi tiến trình và lưu vào size_list.
    size_list = [int(size.item()) for size in size_list] #chuyển kích thước tensor từ LongTensor sang int -> xử lý lúc sau dễ dàng hơn
    max_size = max(size_list) #là kích thước lớn nhất của tất cả các tensor từ các tiến trình.
    # -> các tensor có kích thước khác nhau sẽ cần được đệm để có kích thước bằng nhau trước khi thực hiện all_gather().

   
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(device))
    '''
    Tạo ra một danh sách tensor_list, trong đó mỗi phần tử là một tensor có kích thước max_size.
    '''
    if local_size != max_size: #Nếu kích thước tensor của tiến trình hiện tại khác kích thước lớn nhất
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device) #Tạo ra tensor padding với kích thước bằng max_size - local_size
        tensor = torch.cat((tensor, padding), dim=0) #Nối tensor hiện tại với tensor padding theo chiều 0
    dist.all_gather(tensor_list, tensor) #Thu thập tensor từ tất cả các tiến trình và lưu vào tensor_list

    data_list = [] 
    for size, tensor in zip(size_list, tensor_list): #Duyệt qua từng kích thước và tensor trong size_list và tensor_list
        buffer = tensor.cpu().numpy().tobytes()[:size] # Chuyển tensor -> numpy -> bytes  -> cắt chuỗi byte để lấy đúng kích thước (vì có thể thêm padding)
        data_list.append(pickle.loads(buffer)) #đưa byte về dạng đối tượng python ban đầu

    return data_list #return data_list chứa các đối tượng dữ liệu từ tất cả các tiến trình trog nhóm phân tán


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): tất cả các giá trị sẽ được giảm
        average (bool): có thực hiện trung bình hay tổng
    Giảm các giá trị trong từ điển từ tất cả các tiến trình để tiến trình có rank
    0 có kết quả trung bình. Trả về một dict với các trường giống như
    input_dict, sau khi giảm.
    """
    world_size = get_world_size() #số gpu
    if world_size < 2: # Nếu số gpu < 2 (chỉ có 1) -> ko có GPU khác đang chạy -> return về input_dict
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # Sắp xếp các khóa để chúng nhất quán giữa các tiến trình
        for k in sorted(input_dict.keys()): 
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0) #Các tensor trong values được gộp lại thành 1 tensor duy nhất
        dist.reduce(values, dst=0) #Giảm các tensor từ tất cả các tiến trình về tiến trình có rank là 0 (tiến trình chính).
        if dist.get_rank() == 0 and average:
            # Chỉ tiến trình chính nhận được tích lũy, nên chỉ chia cho
            # world_size trong trường hợp này
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
