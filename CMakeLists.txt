cmake_minimum_required(VERSION 3.15)

# 프로젝트 이름
project(OpenCV_Test)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)

# OpenCV 설정
find_package(OpenCV REQUIRED)

# 실행 파일 설정
add_executable(main src/main.cpp)

# OpenCV 라이브러리 연결
target_link_libraries(main ${OpenCV_LIBS})
