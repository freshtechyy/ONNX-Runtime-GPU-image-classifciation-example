#include "image_classifier.h"

#include <algorithm>

#ifdef VERBOSE
/**
 * @brief Print ONNX tensor type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type) {
  switch (type) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      os << "undefined";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      os << "float";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      os << "uint8_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      os << "int8_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      os << "uint16_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      os << "int16_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      os << "int32_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      os << "int64_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      os << "std::string";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      os << "bool";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      os << "float16";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      os << "double";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      os << "uint32_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      os << "uint64_t";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      os << "float real + float imaginary";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      os << "double real + float imaginary";
      break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      os << "bfloat16";
      break;
    default:
      break;
  }

  return os;
}
#endif

// Constructor
ImageClassifier::ImageClassifier(const std::string& modelFilepath) {
  /**************** Create ORT environment ******************/
  std::string instanceName{"Image classifier inference"};
  mEnv = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                    instanceName.c_str());

  /**************** Create ORT session ******************/
  // Set up options for session
  Ort::SessionOptions sessionOptions;
  // Enable CUDA
  sessionOptions.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
  // Sets graph optimization level (Here, enable all possible optimizations)
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  // Create session by loading the onnx model
  mSession = std::make_shared<Ort::Session>(*mEnv, modelFilepath.c_str(),
                                            sessionOptions);

  /**************** Create allocator ******************/
  // Allocator is used to get model information
  Ort::AllocatorWithDefaultOptions allocator;

  /**************** Input info ******************/
  // Get the number of input nodes
  size_t numInputNodes = mSession->GetInputCount();
#ifdef VERBOSE
  std::cout << "******* Model information below *******" << std::endl;
  std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
#endif

  // Get the name of the input
  // 0 means the first input of the model
  // The example only has one input, so use 0 here
  mInputName = mSession->GetInputName(0, allocator);
#ifdef VERBOSE
  std::cout << "Input Name: " << mInputName << std::endl;
#endif

  // Get the type of the input
  // 0 means the first input of the model
  Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
#ifdef VERBOSE
  std::cout << "Input Type: " << inputType << std::endl;
#endif

  // Get the shape of the input
  mInputDims = inputTensorInfo.GetShape();
#ifdef VERBOSE
  std::cout << "Input Dimensions: " << mInputDims << std::endl;
#endif

  /**************** Output info ******************/
  // Get the number of output nodes
  size_t numOutputNodes = mSession->GetOutputCount();
#ifdef VERBOSE
  std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
#endif

  // Get the name of the output
  // 0 means the first output of the model
  // The example only has one output, so use 0 here
  mOutputName = mSession->GetOutputName(0, allocator);
#ifdef VERBOSE
  std::cout << "Output Name: " << mOutputName << std::endl;
#endif

  // Get the type of the output
  // 0 means the first output of the model
  Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
#ifdef VERBOSE
  std::cout << "Output Type: " << outputType << std::endl;
#endif

  // Get the shape of the output
  mOutputDims = outputTensorInfo.GetShape();
#ifdef VERBOSE
  std::cout << "Output Dimensions: " << mOutputDims << std::endl << std::endl;
#endif
}

// Perform inference for a given image
int ImageClassifier::Inference(const std::string& imageFilepath) {
  // Load an input image
  cv::Mat imageBGR = cv::imread(imageFilepath, cv::IMREAD_COLOR);

  /**************** Preprocessing ******************/
  // Create input tensor (including size and value) from the loaded input image
#ifdef TIME_PROFILE
  const auto before = clock_time::now();
#endif
  // Compute the product of all input dimension
  size_t inputTensorSize = vectorProduct(mInputDims);
  std::vector<float> inputTensorValues(inputTensorSize);
  // Load the image into the inputTensorValues
  CreateTensorFromImage(imageBGR, inputTensorValues);

  // Assign memory for input tensor
  std::vector<const char*> inputNames{mInputName};
  // inputTensors will be used by the Session Run for inference
  std::vector<Ort::Value> inputTensors;
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize, mInputDims.data(),
      mInputDims.size()));

  // Create output tensor (including size and value)
  size_t outputTensorSize = vectorProduct(mOutputDims);
  std::vector<float> outputTensorValues(outputTensorSize);

  // Assign memory for output tensors
  std::vector<const char*> outputNames{mOutputName};
  // outputTensors will be used by the Session Run for inference
  std::vector<Ort::Value> outputTensors;
  outputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, outputTensorValues.data(), outputTensorSize,
      mOutputDims.data(), mOutputDims.size()));

#ifdef TIME_PROFILE
  const sec duration = clock_time::now() - before;
  std::cout << "The preprocessing takes " << duration.count() << "s"
            << std::endl;
#endif

  /**************** Inference ******************/
#ifdef TIME_PROFILE
  const auto before1 = clock_time::now();
#endif
  // 1 means number of inputs and outputs
  // InputTensors and OutputTensors, and inputNames and
  // outputNames are used in Session Run
  mSession->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

#ifdef TIME_PROFILE
  const sec duration1 = clock_time::now() - before1;
  std::cout << "The inference takes " << duration1.count() << "s" << std::endl;
#endif

  /**************** Postprocessing the output result ******************/
#ifdef TIME_PROFILE
  const auto before2 = clock_time::now();
#endif
  // Get the inference result
  float* floatarr = outputTensors.front().GetTensorMutableData<float>();
  // Compute the index of the predicted class
  // 10 means number of classes in total
  int cls_idx = std::max_element(floatarr, floatarr + 10) - floatarr;

#ifdef TIME_PROFILE
  const sec duration2 = clock_time::now() - before2;
  std::cout << "The postprocessing takes " << duration2.count() << "s"
            << std::endl;
#endif

  return cls_idx;
}

// Create a tensor from the input image
void ImageClassifier::CreateTensorFromImage(
    const cv::Mat& img, std::vector<float>& inputTensorValues) {
  cv::Mat imageRGB, scaledImage, preprocessedImage;

  /******* Preprocessing *******/
  // Scale image pixels to [-1, 1]
  img.convertTo(scaledImage, CV_32F, 2.0f / 255.0f, -1.0f);
  // Convert HWC to CHW
  cv::dnn::blobFromImage(scaledImage, preprocessedImage);

  // Assign the input image to the input tensor
  inputTensorValues.assign(preprocessedImage.begin<float>(),
                           preprocessedImage.end<float>());
}
