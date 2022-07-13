#ifndef IMAGE_CLASSIFIER_H_
#define IMAGE_CLASSIFIER_H_

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Header for onnxruntime
#include <onnxruntime_cxx_api.h>

#define VERBOSE
//#define TIME_PROFILE

#ifdef TIME_PROFILE
using clock_time = std::chrono::system_clock;
using sec = std::chrono::duration<double>;
#endif

/**
 * @brief Compute the product over all the elements of a vector
 * @tparam T
 * @param v: input vector
 * @return the product
 */
template <typename T>
size_t vectorProduct(const std::vector<T>& v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

#ifdef VERBOSE
/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}
#endif

class ImageClassifier {
 public:
  /**
   * @brief Constructor
   * @param modelFilepath: path to the .onnx file
   */
  ImageClassifier(const std::string& modelFilepath);

  /**
   * @brief Perform inference on a single image
   * @param imageFilepath: path to the image
   * @return the index of the predicted class
   */
  int Inference(const std::string& imageFilepath);

 private:
  // ORT Environment
  std::shared_ptr<Ort::Env> mEnv;

  // Session
  std::shared_ptr<Ort::Session> mSession;

  // Inputs
  char* mInputName;
  std::vector<int64_t> mInputDims;

  // Outputs
  char* mOutputName;
  std::vector<int64_t> mOutputDims;

  /**
   * @brief Create a tensor from an input image
   * @param img: the input image
   * @param inputTensorValues: the output tensor
   */
  void CreateTensorFromImage(const cv::Mat& img,
                             std::vector<float>& inputTensorValues);
};

#endif  // IMAGE_CLASSIFIER_H_
