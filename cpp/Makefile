SRC = onnx_infer.cpp
COMPILER = g++

# Cross-compilation variables
ifdef CROSS_COMPILE
	COMPILER = $(CROSS_COMPILE)g++
	PKG_CONFIG = $(CROSS_COMPILE)pkg-config
else
	PKG_CONFIG = pkg-config
endif

# Target platform override
ifdef TARGET_PLATFORM
	PLATFORM = $(TARGET_PLATFORM)
else
	PLATFORM = $(shell uname -s)
endif

# check platform and set flags accordingly
ifeq ($(PLATFORM),Linux)
	TARGET = onnx_infer_linux
	ONNX_RUNTIME_DIR = ./onnxruntime-linux-x64-1.23.1
	RPATH = -Wl,-rpath,'$$ORIGIN/$(ONNX_RUNTIME_DIR)/lib' -Wl,-rpath,'$$ORIGIN/$(OPENCV_LIB_DIR)'
	ADDITIONAL_FLAGS = -std=c++17
else ifeq ($(PLATFORM),Darwin)		
	TARGET = onnx_infer_mac
	ONNX_RUNTIME_DIR = ./onnxruntime-osx-arm64-1.23.1
	RPATH = -Wl,-rpath,@executable_path/$(ONNX_RUNTIME_DIR)/lib -Wl,-rpath,@executable_path/$(OPENCV_LIB_DIR)
	ADDITIONAL_FLAGS = -std=c++17
else ifeq ($(PLATFORM),RaspberryPi)
	TARGET = onnx_infer_rpi
	# Note: You'll need to download onnxruntime for ARM/aarch64 and place it here
	ONNX_RUNTIME_DIR = ./onnxruntime-linux-aarch64-1.23.1
	RPATH = -Wl,-rpath,'$$ORIGIN/$(ONNX_RUNTIME_DIR)/lib' -Wl,-rpath,'$$ORIGIN/$(OPENCV_LIB_DIR)'
	ADDITIONAL_FLAGS = -std=c++17 -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard
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
ifeq ($(PLATFORM),Linux)
	@if [ ! -d "$(OPENCV_BUILD_DIR)" ] || [ ! -f "$(OPENCV_LIB_DIR)/libopencv_core.so" ]; then \
		echo "OpenCV not found or not built. Please build OpenCV first with 'make build-opencv'"; \
		exit 1; \
	fi
else ifeq ($(PLATFORM),Darwin)
	@if [ ! -d "$(OPENCV_BUILD_DIR)" ] || [ ! -f "$(OPENCV_LIB_DIR)/libopencv_core.dylib" ]; then \
		echo "OpenCV not found or not built. Please build OpenCV first with 'make build-opencv'"; \
		exit 1; \
	fi
else ifeq ($(PLATFORM),RaspberryPi)
	@if [ ! -d "$(OPENCV_BUILD_DIR)" ] || [ ! -f "$(OPENCV_LIB_DIR)/libopencv_core.so" ]; then \
		echo "OpenCV not found or not built. Please build OpenCV first with 'make build-opencv-rpi'"; \
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

# Build OpenCV for Raspberry Pi (cross-compilation)
build-opencv-rpi:
ifndef CROSS_COMPILE
	@echo "Error: CROSS_COMPILE must be set for Raspberry Pi cross-compilation"
	@echo "Example: make build-opencv-rpi CROSS_COMPILE=aarch64-linux-gnu-"
	@exit 1
endif
	@echo "Building OpenCV for Raspberry Pi..."
	@cd opencv && mkdir -p build && cd build && \
	cmake -DCMAKE_BUILD_TYPE=Release \
		  -DCMAKE_INSTALL_PREFIX=. \
		  -DCMAKE_SYSTEM_NAME=Linux \
		  -DCMAKE_SYSTEM_PROCESSOR=arm \
		  -DCMAKE_C_COMPILER=$(CROSS_COMPILE)gcc \
		  -DCMAKE_CXX_COMPILER=$(CROSS_COMPILE)g++ \
		  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
		  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
		  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
		  -DBUILD_EXAMPLES=OFF \
		  -DBUILD_TESTS=OFF \
		  -DBUILD_PERF_TESTS=OFF \
		  -DWITH_CUDA=OFF \
		  -DWITH_GTK=OFF \
		  -DWITH_QT=OFF \
		  -DENABLE_NEON=ON \
		  -DCMAKE_C_FLAGS="-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard" \
		  -DCMAKE_CXX_FLAGS="-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard" \
		  .. && \
	make -j$(shell nproc 2>/dev/null || echo 4)

$(TARGET): $(SRC)
	$(COMPILER) $^ -o $@ $(INCLUDE) $(LIB) $(ADDITIONAL_FLAGS) $(RPATH)

# Test the build
test: $(TARGET)
	@echo "Testing if the executable can find libraries..."
ifeq ($(PLATFORM),Linux)
	@ldd ./$(TARGET)
else ifeq ($(PLATFORM),Darwin)
	@otool -L ./$(TARGET)
else ifeq ($(PLATFORM),RaspberryPi)
ifdef CROSS_COMPILE
	@echo "Cross-compiled for Raspberry Pi. Testing library dependencies..."
	@$(CROSS_COMPILE)objdump -p ./$(TARGET) | grep NEEDED || echo "No dynamic library dependencies found"
else
	@ldd ./$(TARGET)
endif
endif

# Raspberry Pi specific targets
rpi: TARGET_PLATFORM=RaspberryPi
rpi: $(TARGET)

rpi-cross: 
ifndef CROSS_COMPILE
	@echo "Error: CROSS_COMPILE must be set for cross-compilation"
	@echo "Example: make rpi-cross CROSS_COMPILE=aarch64-linux-gnu-"
	@echo "Or for 32-bit ARM: make rpi-cross CROSS_COMPILE=arm-linux-gnueabihf-"
	@exit 1
endif
	$(MAKE) rpi CROSS_COMPILE=$(CROSS_COMPILE)

clean:
	rm -f onnx_infer_linux onnx_infer_mac onnx_infer_rpi

clean-opencv:
	rm -rf opencv/build

rebuild: clean all

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build for current platform"
	@echo "  rpi              - Build for Raspberry Pi (native compilation)"
	@echo "  rpi-cross        - Cross-compile for Raspberry Pi"
	@echo "  build-opencv     - Build OpenCV for current platform"
	@echo "  build-opencv-rpi - Cross-compile OpenCV for Raspberry Pi"
	@echo "  test             - Test the built executable"
	@echo "  clean            - Remove built executables"
	@echo "  clean-opencv     - Remove OpenCV build directory"
	@echo "  rebuild          - Clean and build"
	@echo ""
	@echo "Cross-compilation examples:"
	@echo "  make rpi-cross CROSS_COMPILE=aarch64-linux-gnu-"
	@echo "  make rpi-cross CROSS_COMPILE=arm-linux-gnueabihf-"
	@echo ""
	@echo "Note: For cross-compilation, you'll need:"
	@echo "  1. Cross-compilation toolchain installed"
	@echo "  2. ONNX Runtime for ARM/aarch64 downloaded to onnxruntime-linux-aarch64-1.23.1/"

.PHONY: all check-opencv build-opencv build-opencv-rpi test clean clean-opencv rebuild rpi rpi-cross help