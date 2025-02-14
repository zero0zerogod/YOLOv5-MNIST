#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

// MNIST 데이터 경로 설정
#define IMAGE_FILE "../data/mnist_data/MNIST/raw/train-images-idx3-ubyte"
#define LABEL_FILE "../data/mnist_data/MNIST/raw/train-labels-idx1-ubyte"

// 이미지 저장 및 YOLO 라벨 생성
void saveImageAndLabel(const Mat &img, int label, int index)
{
    string imgPath = "../data/yolo_data/images/" + to_string(index) + ".jpg";
    imwrite(imgPath, img);

    // 이진화 처리
    Mat bin;
    threshold(img, bin, 10, 255, THRESH_BINARY); // 픽셀 값 10 이상을 흰색(255)으로 설정

    Rect bbox = boundingRect(bin); // 이진화된 이미지에서 숫자 영역 찾기

    //  YOLO 형식 좌표 변환
    float x = (bbox.x + bbox.width / 2.0) / 640.0;
    float y = (bbox.y + bbox.height / 2.0) / 640.0;
    float w = bbox.width / 640.0;
    float h = bbox.height / 640.0;

    // YOLO 레이블 저장
    string labelPath = "../data/yolo_data/labels/" + to_string(index) + ".txt";
    ofstream labelFile(labelPath);
    labelFile << label << " ";
    labelFile << fixed << setprecision(6) << x << " " << y << " " << w << " " << h << endl;
    labelFile.close();
}

int main()
{
    ifstream imageFile(IMAGE_FILE, ios::binary);
    ifstream labelFile(LABEL_FILE, ios::binary);

    if (!imageFile.is_open() || !labelFile.is_open())
    {
        cerr << "Could not open MNIST dataset!" << endl;
        return -1;
    }

    // MNIST 데이터셋 헤더 건너뛰기
    imageFile.seekg(16, ios::beg);
    labelFile.seekg(8, ios::beg);

    for (int i = 0; i < 60000; i++)
    { // 60,000개의 MNIST 학습 데이터
        Mat img(28, 28, CV_8U);
        char label;
        imageFile.read((char *)img.data, 28 * 28);
        labelFile.read(&label, 1);

        resize(img, img, Size(640, 640));
        saveImageAndLabel(img, label, i);
    }

    imageFile.close();
    labelFile.close();
    return 0;
}
