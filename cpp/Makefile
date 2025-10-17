SRC = onnx_infer.cpp
COMPILER = g++


# check platform and set flags accordingly
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	TARGET = onnx_infer_linux
	ONNX_RUNTIME_DIR = ./onnxruntime-linux-x64-1.23.1
	RPATH = -Wl,-rpath,'$$ORIGIN/$(ONNX_RUNTIME_DIR)/lib' -Wl,-rpath,'$$ORIGIN/$(OPENCV_LIB_DIR)'

	ADDITIONAL_FLAGS = -std=c++17
else ifeq ($(UNAME_S),Darwin)		
	TARGET = onnx_infer_mac
	ONNX_RUNTIME_DIR = ./onnxruntime-osx-arm64-1.23.1
	RPATH = -Wl,-rpath,@executable_path/$(ONNX_RUNTIME_DIR)/lib -Wl,-rpath,@executable_path/$(OPENCV_LIB_DIR)

	ADDITIONAL_FLAGS = -std=c++17
endif

# OpenCV configuration
OPENCV_BUILD_DIR = ./opencv/build
OPENCV_SOURCE_DIR = ./opencv
OPENCV_INCLUDE = -I$(OPENCV_SOURCE_DIR)/include -I$(OPENCV_BUILD_DIR)
OPENCV_INCLUDE += -I$(OPENCV_SOURCE_DIR)/modules/core/include
OPENCV_INCLUDE += -I$(OPENCV_SOURCE_DIR)/modules/imgproc/include
OPENCV_INCLUDE += -I$(OPENCV_SOURCE_DIR)/modules/imgcodecs/include
OPENCV_INCLUDE += -I$(OPENCV_SOURCE_DIR)/modules/highgui/include
OPENCV_INCLUDE += -I$(OPENCV_SOURCE_DIR)/modules/dnn/include
OPENCV_LIB_DIR = $(OPENCV_BUILD_DIR)/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_dnn

# Include paths
INCLUDE = -I$(ONNX_RUNTIME_DIR)/include
INCLUDE += $(OPENCV_INCLUDE)

# Library paths and links
LIB = -L$(ONNX_RUNTIME_DIR)/lib -lonnxruntime
LIB += -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS)

# Default target
all: check-opencv $(TARGET)

# Check if OpenCV is built
check-opencv:
ifeq ($(UNAME_S),Linux)
	@if [ ! -d "$(OPENCV_BUILD_DIR)" ] || [ ! -f "$(OPENCV_LIB_DIR)/libopencv_core.so" ]; then \
		echo "OpenCV not found or not built. Please build OpenCV first with 'make build-opencv'"; \
		exit 1; \
	fi
else ifeq ($(UNAME_S),Darwin)
	@if [ ! -d "$(OPENCV_BUILD_DIR)" ] || [ ! -f "$(OPENCV_LIB_DIR)/libopencv_core.dylib" ]; then \
		echo "OpenCV not found or not built. Please build OpenCV first with 'make build-opencv'"; \
		exit 1; \
	fi
endif

# Build OpenCV (if needed)
build-opencv:
	@echo "Building OpenCV..."
	@cd opencv && mkdir -p build && cd build && \
	cmake -DCMAKE_BUILD_TYPE=Release \
		  -DCMAKE_INSTALL_PREFIX=. \
		  -DBUILD_EXAMPLES=OFF \
		  -DBUILD_TESTS=OFF \
		  -DBUILD_PERF_TESTS=OFF \
		  -DWITH_CUDA=OFF \
		  .. && \
	make -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

$(TARGET): $(SRC)
	$(COMPILER) $^ -o $@ $(INCLUDE) $(LIB) $(ADDITIONAL_FLAGS) $(RPATH)

# Test the build
test: $(TARGET)
	@echo "Testing if the executable can find libraries..."
ifeq ($(UNAME_S),Linux)
	@ldd ./$(TARGET)
else ifeq ($(UNAME_S),Darwin)
	@otool -L ./$(TARGET)
endif

clean:
	rm -f $(TARGET)

clean-opencv:
	rm -rf opencv/build

rebuild: clean all

.PHONY: all check-opencv build-opencv test clean clean-opencv rebuild