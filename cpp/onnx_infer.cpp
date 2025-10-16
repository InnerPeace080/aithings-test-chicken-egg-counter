
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

int main(int argc, char* argv[]) {
  Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;

  const std::string exe_path = get_executable_dir();
  Ort::Session      session(env, (exe_path + "/../models/yolov8n_chicken_egg.onnx").c_str(), session_options);

  return 0;
}