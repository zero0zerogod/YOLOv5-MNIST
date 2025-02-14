import os
import shutil
import random

# 데이터 경로 설정
ROOT_DIR = "../data/yolo_data"
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
LABEL_DIR = os.path.join(ROOT_DIR, "labels")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
VAL_DIR = os.path.join(ROOT_DIR, "val")
TEST_DIR = os.path.join(ROOT_DIR, "test") 

# train/val/test 디렉터리 생성
for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)

# 이미지 파일 리스트 로드
random.seed(42)  # 항상 같은 데이터 분할 유지
image_files = sorted(os.listdir(IMAGE_DIR))
random.shuffle(image_files)

# 70% train / 20% val / 10% test로 분할
train_split = int(len(image_files) * 0.7)
val_split = int(len(image_files) * 0.9)  # val까지 포함한 인덱스

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]  # 나머지 10%는 test

# train/val/test 디렉터리로 데이터 복사
for files, folder in [(train_files, TRAIN_DIR), (val_files, VAL_DIR), (test_files, TEST_DIR)]:
    for file in files:
        src_img = os.path.join(IMAGE_DIR, file)
        dst_img = os.path.join(folder, "images", file)
        
        src_label = os.path.join(LABEL_DIR, file.replace(".jpg", ".txt"))
        dst_label = os.path.join(folder, "labels", file.replace(".jpg", ".txt"))

        # 파일이 존재하면 덮어쓰지 않고 삭제 후 복사
        if os.path.exists(dst_img):
            os.remove(dst_img)
        if os.path.exists(dst_label):
            os.remove(dst_label)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)

print("Dataset successfully split into train, val, and test sets.")
