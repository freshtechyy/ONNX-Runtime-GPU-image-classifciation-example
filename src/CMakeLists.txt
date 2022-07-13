cmake_minimum_required(VERSION 3.20)

project(ONNX_Runtime_Depth_Estimation_EXAMPLE)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(image_classifier main.cpp image_classifier.cpp)
target_include_directories(image_classifier PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_classifier PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBRARIES})
