
#if defined(__linux__)
#include <linux/limits.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <onnxruntime_cxx_api.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>

std::string get_executable_dir() {
  // for linux and darwin
#if defined(__linux__)
  char    result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count != -1) {
    std::string exec_path(result, count);
    size_t      last_slash = exec_path.find_last_of('/');
    if (last_slash != std::string::npos) {
      return exec_path.substr(0, last_slash);
    }
  }

#elif defined(__APPLE__)
  char     result[PATH_MAX];
  uint32_t size = sizeof(result);
  if (_NSGetExecutablePath(result, &size) == 0) {
    std::string exec_path(result);
    size_t      last_slash = exec_path.find_last_of('/');
    if (last_slash != std::string::npos) {
      return exec_path.substr(0, last_slash);
    }
  }
#endif
  return "";
}

int main(int argc, char *argv[]) {
  Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;

  // List available providers for debugging
  auto available_providers = Ort::GetAvailableProviders();
  std::cout << "Available execution providers: ";
  for (const auto &provider : available_providers) {
    std::cout << provider << " ";
  }
  std::cout << std::endl;

  // trace process time
  auto start_time = std::chrono::high_resolution_clock::now();

  const std::string exe_path = get_executable_dir();
  Ort::Session      session(env, (exe_path + "/../models/yolo_chicken_egg_infer.quant.onnx").c_str(), session_options);

  // time to load model
  std::cout << "Model loaded in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
                   .count()
            << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
    return -1;
  }

  std::string        input_image = argv[1];
  std::vector<float> input_tensor_values;

  // read image file, resize to 640x640, RBG, normalize to [0,1], convert to float32
  cv::Mat image = cv::imread(input_image);

  std::cout << "Image read in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
                   .count()
            << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();

  cv::Mat original_image  = image.clone();
  int     original_width  = image.cols;
  int     original_height = image.rows;

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::resize(image, image, cv::Size(640, 640));

  std::cout << "Image preprocessed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
                   .count()
            << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();

  // check shape of image
  std::cout << "Image shape: " << image.cols << " " << image.rows << " " << image.channels() << std::endl;
  // convert HWC to CHW
  std::vector<cv::Mat> channels(3);
  cv::split(image, channels);

  // copy channels to input_tensor_values one by one channel
  for (int c = 0; c < 3; ++c) {
    cv::Mat channel = channels[c];
    for (int i = 0; i < channel.rows; ++i) {
      for (int j = 0; j < channel.cols; ++j) {
        input_tensor_values.push_back(static_cast<float>(channel.at<uchar>(i, j)) / 255.0);
      }
    }
  }

  std::cout << "Image converted to tensor in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
                   .count()
            << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();

  const auto  input_names       = session.GetInputNames();
  const char *input_name        = input_names[0].c_str();
  const auto  input_type_info   = session.GetInputTypeInfo(0);
  const auto  input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
  const auto  input_shape       = input_tensor_info.GetShape();
  std::cout << "Input shape: ";
  for (const auto &dim : input_shape) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  const auto  output_names       = session.GetOutputNames();
  const char *output_name        = output_names[0].c_str();
  const auto  output_type_info   = session.GetOutputTypeInfo(0);
  const auto  output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::Value input_tensor      = Ort::Value::CreateTensor<float>(allocator, input_shape.data(), input_shape.size());
  float     *input_tensor_data = input_tensor.GetTensorMutableData<float>();
  // write input_tensor_values data to input_tensor_data
  std::memcpy(input_tensor_data, input_tensor_values.data(), input_tensor_values.size() * sizeof(float));

  std::cout << "Input tensor created in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
                   .count()
            << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();

  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

  std::cout << "Inference done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
                   .count()
            << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();

  float     *output_data  = output_tensors.front().GetTensorMutableData<float>();
  const auto output_shape = output_tensor_info.GetShape();
  std::cout << "Output shape: ";
  for (const auto &dim : output_shape) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  // output_data shape (1 6 8400)
  // Assume output_data is [1, 6, 8400]: [batch, attributes, num_boxes]

  int num_boxes = output_shape[2];
  std::cout << "Number of boxes: " << num_boxes << std::endl;
  // transform output_data to [8400, 6]
  std::vector<std::array<float, 6>> boxes_attrs(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    for (int j = 0; j < 6; ++j) {
      boxes_attrs[i][j] = output_data[j * num_boxes + i];
    }
  }

  std::vector<cv::Rect> boxes;
  std::vector<float>    scores;
  std::vector<int>      class_ids;
  float                 conf_threshold = 0.591f;
  float                 nms_threshold  = 0.4f;

  int max_score = 0;

  for (int i = 0; i < num_boxes; ++i) {
    float conf_cls0 = boxes_attrs[i][4];
    float conf_cls1 = boxes_attrs[i][5];

    float score = std::max(conf_cls0, conf_cls1);
    max_score   = std::max(max_score, static_cast<int>(score * 100));

    float x1 = (boxes_attrs[i][0] - boxes_attrs[i][2] / 2);
    float y1 = (boxes_attrs[i][1] - boxes_attrs[i][3] / 2);
    float w  = boxes_attrs[i][2];
    float h  = boxes_attrs[i][3];
    boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(w), static_cast<int>(h));
    scores.push_back(score);
    class_ids.push_back(conf_cls0 > conf_cls1 ? 0 : 1);
  }

  std::cout << "Boxes and scores extracted in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
                   .count()
            << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);
  std::cout << "Detections after NMS: " << indices.size() << std::endl;
  for (int idx : indices) {
    cv::Rect box      = boxes[idx];
    int      class_id = class_ids[idx];
    // scale box to original image size
    int      ori_x      = static_cast<int>(box.x * original_width / 640.0);
    int      ori_y      = static_cast<int>(box.y * original_height / 640.0);
    int      ori_width  = static_cast<int>(box.width * original_width / 640.0);
    int      ori_height = static_cast<int>(box.height * original_height / 640.0);
    cv::Rect ori_box    = cv::Rect(ori_x, ori_y, ori_width, ori_height);

    std::cout << "Box: [" << ori_x << ", " << ori_y << ", " << ori_width << ", " << ori_height
              << "] Score: " << scores[idx] << ", Class ID: " << class_id << std::endl;
  }

  return 0;
}