"""
Tác dụng chính của file này:
- Cung cấp các hàm để đánh giá độ chính xác và độ phủ cho việc định vị vi phạm video ở cấp độ phân đoạn.
- Bao gồm các hàm tính toán độ dài đoạn, phần giao giữa các hộp, độ chính xác và độ phủ.
- Cung cấp hàm đánh giá kết quả trên toàn bộ tập dữ liệu video.

Giải thích các hàm chính:
- seg_len_accumulate và seg_len: Tính tổng độ dài của các đoạn, hỗ trợ tính hợp hoặc tổng.
- calc_inter: Tính phần giao giữa các hộp dự đoán và hộp ground truth.
- precision_recall: Tính độ chính xác và độ phủ cho một video.
- evaluate: Đánh giá kết quả trên toàn bộ tập dữ liệu video, tính độ phủ và độ chính xác trung bình.

Các hàm này rất quan trọng trong việc đánh giá hiệu suất của mô hình định vị vi phạm video,
giúp so sánh kết quả dự đoán với ground truth và đưa ra các chỉ số đánh giá cụ thể.
"""

from typing import Dict, Any

import numpy as np

"""
Các hàm đánh giá độ chính xác/độ phủ cho việc định vị vi phạm video ở cấp độ phân đoạn
"""

def seg_len_accumulate(segments: np.ndarray, type: str = 'union') -> int:
    """
    Tính tổng độ dài của tất cả các đoạn
    
    Tham số:
    segments: mảng numpy chứa các đoạn, mỗi hàng là một đoạn (start, end)
    type: 'union' - tính hợp của các đoạn, 'sum' - tính tổng độ dài các đoạn
    
    Trả về: Tổng độ dài các đoạn
    """
    init_score = np.zeros(segments[:, 1].max(), dtype=int)
    for seg in segments:
        init_score[seg[0]: seg[1]] += 1
    if type == 'union':
        return sum(init_score.astype(bool))
    else:
        return sum(init_score)


def seg_len(segments: np.ndarray, type: str = 'union') -> float:
    """
    Tính tổng độ dài của tất cả các đoạn
    
    Tham số:
    segments: mảng numpy chứa các đoạn, mỗi hàng là một đoạn (start, end)
    type: 'union' - tính hợp của các đoạn, 'sum' - tính tổng độ dài các đoạn
    
    Trả về: Tổng độ dài các đoạn
    """

    if type != 'union':
        return np.sum(segments[:, 1] - segments[:, 0]).item()

    segments_to_sum = []
    segments = sorted(segments.tolist(), key=lambda x: x[0])
    for segment in segments:
        if len(segments_to_sum) == 0:
            segments_to_sum.append(segment)
            continue

        last_segment = segments_to_sum[-1]
        if last_segment[1] < segment[0]:
            segments_to_sum.append(segment)
        else:
            union_segment = [min(last_segment[0], segment[0]), max(last_segment[1], segment[1])]
            segments_to_sum[-1] = union_segment

    segments_to_sum = np.array(segments_to_sum, dtype=np.float32)
    return np.sum(segments_to_sum[:, 1] - segments_to_sum[:, 0]).item()


def calc_inter(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Tính phần giao giữa các hộp dự đoán và hộp ground truth
    
    Tham số:
    pred_boxes: mảng numpy chứa các hộp dự đoán
    gt_boxes: mảng numpy chứa các hộp ground truth
    
    Trả về: 
    inter_boxes: mảng numpy chứa các hộp giao
    inter_areas: mảng numpy chứa diện tích giao
    """
    lt = np.maximum(pred_boxes[:, None, :2], gt_boxes[:, :2])
    rb = np.minimum(pred_boxes[:, None, 2:], gt_boxes[:, 2:])
    wh = np.maximum(rb - lt, 0)
    inter_boxes = np.concatenate((lt, rb), axis=2)
    inter_areas = wh[:, :, 0] * wh[:, :, 1]
    return inter_boxes, inter_areas


def precision_recall(pred_boxes: np.ndarray, gt_boxes: np.ndarray):
    """
    Tính độ chính xác và độ phủ
    
    Tham số:
    pred_boxes: mảng numpy chứa các hộp dự đoán
    gt_boxes: mảng numpy chứa các hộp ground truth
    
    Trả về: Dict chứa độ chính xác và độ phủ
    """
    if len(pred_boxes) > 0 and len(gt_boxes) == 0:
        return {"precision": 0, "recall": 1}

    if len(pred_boxes) == 0 and len(gt_boxes) > 0:
        return {"precision": 1, "recall": 0}

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {"precision": 1, "recall": 1}

    inter_boxes, inter_areas = calc_inter(pred_boxes, gt_boxes)

    sum_tp_w, sum_p_w, sum_tp_h, sum_p_h = 0, 0, 0, 0
    for pred_ind, inter_per_pred in enumerate(inter_areas):
        pos_gt_inds = np.where(inter_per_pred > 0)
        if len(pos_gt_inds[0]) > 0:
            sum_tp_w += seg_len(np.squeeze(inter_boxes[pred_ind, pos_gt_inds, :][:, :, [0, 2]], axis=0))
            sum_tp_h += seg_len(np.squeeze(inter_boxes[pred_ind, pos_gt_inds, :][:, :, [1, 3]], axis=0))

    sum_p_w = seg_len(pred_boxes[:, [0, 2]], type='sum')
    sum_p_h = seg_len(pred_boxes[:, [1, 3]], type='sum')
    precision_w = sum_tp_w / (sum_p_w + 1e-6)
    precision_h = sum_tp_h / (sum_p_h + 1e-6)

    sum_tp_w, sum_p_w, sum_tp_h, sum_p_h = 0, 0, 0, 0
    for gt_ind, inter_per_gt in enumerate(inter_areas.T):
        pos_pred_inds = np.where(inter_per_gt > 0)
        if len(pos_pred_inds[0]) > 0:
            sum_tp_w += seg_len(np.squeeze(inter_boxes[pos_pred_inds, gt_ind, :][:, :, [0, 2]], axis=0))
            sum_tp_h += seg_len(np.squeeze(inter_boxes[pos_pred_inds, gt_ind, :][:, :, [1, 3]], axis=0))

    sum_p_w = seg_len(gt_boxes[:, [0, 2]], type='sum')
    sum_p_h = seg_len(gt_boxes[:, [1, 3]], type='sum')
    recall_w = sum_tp_w / (sum_p_w + 1e-6)
    recall_h = sum_tp_h / (sum_p_h + 1e-6)

    return {"precision": precision_h * precision_w, "recall": recall_h * recall_w}


def evaluate(result_dict: Dict[str, Dict[str, Any]], video_set_dict: Dict[str, Any]):
    """
    Đánh giá kết quả trên toàn bộ tập dữ liệu
    
    Tham số:
    result_dict: Dict chứa kết quả cho từng video
    video_set_dict: Dict chứa thông tin về tập video
    
    Trả về: Độ phủ trung bình, độ chính xác trung bình và số lượng video
    """
    macro_result_list = []
    for video_id in video_set_dict:
        precision_list = [result_dict[i]['precision'] for i in video_set_dict[video_id]]
        recall_list = [result_dict[i]['recall'] for i in video_set_dict[video_id]]
        r, p = sum(recall_list)/len(recall_list), sum(precision_list)/len(precision_list)
        macro_result = (r, p, )
        macro_result_list.append(macro_result)

    r, p = map(sum, zip(*macro_result_list))
    cnt = len(macro_result_list)
    r, p = r / cnt, p / cnt
    return r, p, cnt
