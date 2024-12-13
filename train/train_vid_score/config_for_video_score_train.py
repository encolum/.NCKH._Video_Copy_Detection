root_dir= "D:\VideoMatching"

img_width=224

feat_dim = 512
bert_dim = 768
gradient_checkpointing = False

max_frames = 256
output_dim = 256

bert_path = r"D:\VideoMatching\checkpoints_train_vid_score\googlebert-base"
data = dict(
    # Cấu hình cho tập dữ liệu huấn luyện
    train=dict(
        # Loại dataset sử dụng
        type="LabelFeatZipDataSet",
        # Đường dẫn đến file chứa ID của các video truy vấn
        vids_path=f"{root_dir}/data/meta/train/train_query_id.csv",  
        # Đường dẫn đến file zip chứa các đặc trưng của video
        feat_zip_path=r"D:\VideoMatching\data\feat_zips\feats1.zip", #TODO
        # Số khung hình tối đa trong một video
        max_frames=256,
        # Số lượng worker để tải dữ liệu
        num_workers=8,
        # Đường dẫn đến file chứa các truy vấn tích cực
        ann_vids_path=f"{root_dir}/data/meta/train/train_positive_query.txt"
    ),
    # Cấu hình cho tập dữ liệu đánh giá
    val=dict(
        # Loại dataset sử dụng
        type="LabelFeatZipDataSet",
        # Đường dẫn đến file chứa ID của các video truy vấn
        vids_path=f"{root_dir}/data/meta/val/val_query_id.csv",
        # Đường dẫn đến file zip chứa các đặc trưng của video
        feat_zip_path=r"D:\VideoMatching\data\feat_zips\feats1.zip", #TODO
        # Số khung hình tối đa trong một video
        max_frames=256,
        # Số lượng worker để tải dữ liệu
        num_workers=8,
        # Đường dẫn đến file chứa các truy vấn tích cực
        ann_vids_path=f"{root_dir}/data/meta/train/train_positive_query.txt"
        ) 
)    