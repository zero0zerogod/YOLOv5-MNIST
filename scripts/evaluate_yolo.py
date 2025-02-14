import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLOv5 디렉터리
YOLOV5_DIR = os.path.abspath(os.path.join(BASE_DIR, "../yolov5"))
if not os.path.exists(YOLOV5_DIR):
    raise FileNotFoundError(f"YOLOv5 directory not found at: {YOLOV5_DIR}")

# 모델(학습 완료된) 경로
model_path = os.path.abspath(os.path.join(BASE_DIR, "../models/mnist.pt"))
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model does not exist: {model_path}")

# data yaml
data_yaml = os.path.abspath(os.path.join(BASE_DIR, "../data/yolo_data/yolo_mnist.yaml"))
if not os.path.exists(data_yaml):
    raise FileNotFoundError(f"Data config does not exist: {data_yaml}")

eval_results_path = os.path.abspath(os.path.join(BASE_DIR, "../models/evaluation_results.txt"))

# YOLOv5 val.py 실행 명령어
val_command = [
    sys.executable,               # 파이썬 실행기
    "val.py",                     # yolov5/val.py 스크립트
    "--data", data_yaml,
    "--weights", model_path,
    "--batch", "8",
    "--img", "640",
    "--task", "test"              # optional: val/test
]

print("\nStarting YOLOv5 evaluation...\n" + "=" * 50)

# YOLOv5 디렉터리에서 val.py 실행
process = subprocess.Popen(
    val_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    cwd=YOLOV5_DIR  # cwd를 YOLOv5 폴더로 설정
)

output_lines = []
for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
    output_lines.append(line)

process.wait()

if process.returncode != 0:
    full_output = "".join(output_lines)
    print(f"\nError during evaluation. Full log output:\n{full_output}\n")
    sys.exit(1)

print("\nEvaluation completed successfully.")
print(f"Results are saved in YOLOv5 default path (runs/val/*).")

# 필요하면 결과를 저장(추가 기능)
with open(eval_results_path, "w") as f:
    f.write("".join(output_lines))

print(f"\nEvaluation log saved to {eval_results_path}.\n")
