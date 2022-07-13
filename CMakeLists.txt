cmake_minimum_required(VERSION 3.20)

project(ONNX_Runtime_Depth_Estimation)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# TODO: Change the path to your ONNX Runtime installation directory
set(ONNXRUNTIME_ROOT_PATH ~/onnx/)
set(ONNXRUNTIME_INCLUDE_DIRS
  ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/include/onnxruntime
  ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
  ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/include/onnxruntime/core/session
)

set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/onnxruntime/build/Linux/Release/libonnxruntime.so)

find_package(OpenCV REQUIRED)

add_subdirectory(src)
