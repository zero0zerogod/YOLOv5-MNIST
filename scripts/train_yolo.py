import os
import sys
import shutil
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 로컬 YOLOv5 경로
YOLOV5_DIR = os.path.abspath(os.path.join(BASE_DIR, "../yolov5"))
if not os.path.exists(YOLOV5_DIR):
    raise FileNotFoundError(f"YOLOv5 directory not found at: {YOLOV5_DIR}")

# 모델/체크포인트 관련 경로
weights_path = os.path.abspath(os.path.join(BASE_DIR, "../models/yolov5s.pt"))
output_dir = os.path.abspath(os.path.join(BASE_DIR, "../runs/train/exp"))
final_model_path = os.path.abspath(os.path.join(BASE_DIR, "../models/mnist.pt"))

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Pretrained YOLOv5 model does not exist: {weights_path}")

# 데이터 yaml
data_yaml = os.path.abspath(os.path.join(BASE_DIR, "../data/yolo_data/yolo_mnist.yaml"))
project_dir = os.path.abspath(os.path.join(BASE_DIR, "../runs/train"))

# 체크포인트 경로
resume_checkpoint = os.path.join(output_dir, "weights", "last.pt")
resume_flag = os.path.exists(resume_checkpoint) and os.path.isfile(resume_checkpoint)
if resume_flag:
    print(f"Checkpoint detected: {resume_checkpoint}. Resuming training.")
else:
    print("No valid checkpoint detected. Starting training from scratch.")

# base train command
train_command = [
    sys.executable,
    os.path.join(YOLOV5_DIR, "train.py"),
    "--img", "640",
    "--batch", "8",
    "--epochs", "20",
    "--data", data_yaml,
    "--device", "0",
    "--project", project_dir,
    "--name", "exp",
]

# resume 플래그 처리
if resume_flag:
    # resume 시, last.pt 경로를 직접 넘김
    # => YOLOv5가 이 경로를 정확히 체크포인트로 인식하여 기존 상태를 불러옴
    train_command.append(f"--resume")
    train_command.append(os.path.abspath(resume_checkpoint))
else:
    # 처음 학습 시, 사전 학습 weights를 지정
    train_command.extend(["--weights", weights_path])

print("\nStarting YOLOv5 training...\n" + "=" * 50)
print("DEBUG: train_command =", train_command)
sys.stdout.flush()

# YOLOv5 디렉터리를 working directory로
process = subprocess.Popen(
    train_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    cwd=YOLOV5_DIR
)

output_lines = []
for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
    output_lines.append(line)

process.wait()

if process.returncode != 0:
    full_output = "".join(output_lines)
    print(f"\nError during training. Full log output:\n{full_output}\n")
    sys.exit(1)

print("\nTraining completed successfully.")

# best.pt -> 최종 모델 저장
best_model_path = os.path.join(output_dir, "weights", "best.pt")
if os.path.exists(best_model_path):
    shutil.copy(best_model_path, final_model_path)
    print(f"\nTraining completed. Model saved as {final_model_path}\n")
else:
    print("\nWarning: best.pt not found. Please check if training completed successfully.\n")
