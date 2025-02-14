import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 main.py가 위치한 디렉터리
SCRIPTS_DIR = os.path.join(BASE_DIR, "../scripts")      # scripts 디렉터리 절대 경로

# 실행할 스크립트 리스트
scripts = [
    "split_data.py",      # 데이터 분할
    "train_yolo.py",      # YOLOv5 학습
    "evaluate_yolo.py",   # 모델 평가
    "export_model.py"     # ONNX 변환
]

# 모든 스크립트 실행
for script in scripts:
    script_path = os.path.join(SCRIPTS_DIR, script)
    print(f"\nRunning: {script_path}\n" + "=" * 50)
    
    # stderr와 stdout을 합쳐서 처리
    process = subprocess.Popen(
        ["python3", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # 표준 출력을 실시간으로 출력
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"\nError occurred while running: {script_path}\n")
        break  # 오류 발생 시 중단

print("\nFull pipeline execution completed.\n")
