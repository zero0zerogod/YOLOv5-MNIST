import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV5_DIR = os.path.abspath(os.path.join(BASE_DIR, "../yolov5"))

if not os.path.exists(YOLOV5_DIR):
    raise FileNotFoundError(f"YOLOv5 directory not found at: {YOLOV5_DIR}")

# 모델 경로 및 ONNX 출력 파일 경로
model_path = os.path.abspath(os.path.join(BASE_DIR, "../models/mnist.pt"))
onnx_output = os.path.abspath(os.path.join(BASE_DIR, "../models/mnist.onnx"))

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model does not exist: {model_path}")

# `export.py` 실행 명령어
export_command = [
    sys.executable,
    os.path.join(YOLOV5_DIR, "export.py"),
    "--weights", model_path,
    "--simplify",
    "--dynamic",
    "--include", "onnx"
]

print("\nStarting YOLOv5 model conversion to ONNX...\n" + "=" * 50)
sys.stdout.flush()

# subprocess 실행
process = subprocess.Popen(
    export_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    cwd=YOLOV5_DIR  # YOLOv5 디렉터리에서 실행
)

for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

process.wait()

if process.returncode != 0:
    print("\nError occurred during ONNX conversion.\n")
    sys.exit(1)

# 변환된 ONNX 파일이 존재하는지 확인
if os.path.exists(onnx_output):
    print(f"\nONNX conversion completed: {onnx_output}\n")
else:
    print("\nONNX conversion failed.\n")
