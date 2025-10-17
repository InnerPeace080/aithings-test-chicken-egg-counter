SRC = onnx_infer.cpp
COMPILER = g++

# Cross-compilation variables
ifdef CROSS_COMPILE
	COMPILER = $(CROSS_COMPILE)g++
	PKG_CONFIG = $(CROSS_COMPILE)pkg-config
else
	PKG_CONFIG = pkg-config
endif

# Target platform override - use command line or environment variable
PLATFORM ?= $(shell uname -s)

# Platform-specific settings
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
	# Compatibility flags for older glibc (Raspberry Pi typically has older versions)
	ADDITIONAL_FLAGS = -std=c++17 -march=armv8-a -static-libgcc -static-libstdc++
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

# Detect OpenCV library type (static vs shared) and set appropriate linking flags
ifeq ($(PLATFORM),RaspberryPi)
	# Check if static libraries exist first (built by Docker)
	ifneq ($(wildcard $(OPENCV_LIB_DIR)/libopencv_core.a),)
		# Static libraries found - use them for maximum compatibility
		OPENCV_LIBS = -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
	else
		# Use shared libraries (default)
		OPENCV_LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_dnn
	endif
else
	# For non-RaspberryPi platforms, use shared libraries
	OPENCV_LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_dnn
endif

# Include paths
INCLUDE = -I$(ONNX_RUNTIME_DIR)/include
INCLUDE += $(OPENCV_INCLUDE)

# Library paths and links
LIB = -L$(ONNX_RUNTIME_DIR)/lib -lonnxruntime
LIB += -L$(OPENCV_LIB_DIR) $(OPENCV_LIBS)

# Additional static libraries needed for ARM64 cross-compilation
ifeq ($(PLATFORM),RaspberryPi)
	OPENCV_3RDPARTY_LIBS = -L./opencv/build/3rdparty/lib -llibprotobuf -littnotify -lade -llibjpeg-turbo -llibwebp -llibpng -llibtiff -llibopenjp2 -lzlib -ltegra_hal
	STATIC_LINK_FLAGS = -Wl,--allow-shlib-undefined -ldl -lpthread
else
	OPENCV_3RDPARTY_LIBS =
	STATIC_LINK_FLAGS =
endif

# Default target
all: check-opencv $(TARGET)

# Check if OpenCV is built (for native builds only)
check-opencv:
ifeq ($(PLATFORM),Linux)
	@if [ ! -d "$(OPENCV_BUILD_DIR)" ] || [ ! -f "$(OPENCV_LIB_DIR)/libopencv_core.so" ]; then \
		echo "OpenCV not found. Build with 'make build-opencv-native' or use 'make rpi-docker' for Raspberry Pi"; \
		exit 1; \
	fi
else ifeq ($(PLATFORM),Darwin)
	@if [ ! -d "$(OPENCV_BUILD_DIR)" ] || [ ! -f "$(OPENCV_LIB_DIR)/libopencv_core.dylib" ]; then \
		echo "OpenCV not found. Build with 'make build-opencv-native'"; \
		exit 1; \
	fi
else ifeq ($(PLATFORM),RaspberryPi)
	@if [ ! -d "$(OPENCV_BUILD_DIR)" ]; then \
		echo "OpenCV build directory not found. Use 'make rpi-docker' to build"; \
		exit 1; \
	elif [ -f "$(OPENCV_LIB_DIR)/libopencv_core.a" ]; then \
		echo "Found OpenCV static libraries"; \
	elif [ -f "$(OPENCV_LIB_DIR)/libopencv_core.so" ]; then \
		echo "Found OpenCV shared libraries"; \
	else \
		echo "OpenCV libraries not found. Use 'make rpi-docker' to build"; \
		exit 1; \
	fi
endif



# Build OpenCV natively (for local development only)
build-opencv-native:
	@echo "Building OpenCV natively..."
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
	@echo "Building $(TARGET) for $(PLATFORM)..."
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

# Native compilation on Raspberry Pi
rpi-native: 
	@echo "Building natively on Raspberry Pi..."
	@if [ "$$(uname -m)" != "aarch64" ] && [ "$$(uname -m)" != "armv7l" ]; then \
		echo "❌ This target should be run on a Raspberry Pi"; \
		echo "💡 For cross-compilation, use: make rpi-docker"; \
		exit 1; \
	fi
	$(MAKE) all PLATFORM=RaspberryPi

# Build for Raspberry Pi using Docker (MAIN TARGET)
rpi-docker:
	@echo "Building for Raspberry Pi using Docker with compatible glibc..."
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "❌ Docker not found. Please install Docker first:"; \
		echo "   sudo apt update && sudo apt install docker.io"; \
		echo "   sudo usermod -aG docker $$USER  # Then logout and login"; \
		exit 1; \
	fi
	@echo "🐳 Starting Docker build with Debian 11 (glibc 2.31)..."
	docker run --rm -v $(PWD):/workspace -w /workspace \
		debian:11 /bin/bash -c " \
		set -e && \
		echo '📦 Installing build dependencies...' && \
		apt-get update -qq && \
		apt-get install -y -qq build-essential cmake git wget && \
		apt-get install -y -qq gcc-aarch64-linux-gnu g++-aarch64-linux-gnu && \
		apt-get install -y -qq libjpeg-dev libpng-dev libtiff-dev zlib1g-dev && \
		echo '🧹 Cleaning previous builds...' && \
		rm -rf opencv/build && \
		echo '🔨 Building OpenCV for ARM64...' && \
		cd opencv && mkdir -p build && cd build && \
		cmake -DCMAKE_BUILD_TYPE=Release \
			  -DCMAKE_INSTALL_PREFIX=. \
			  -DCMAKE_SYSTEM_NAME=Linux \
			  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
			  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
			  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
			  -DCMAKE_C_FLAGS='-march=armv8-a' \
			  -DCMAKE_CXX_FLAGS='-march=armv8-a' \
			  -DBUILD_SHARED_LIBS=OFF \
			  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
			  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
			  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
			  -DBUILD_EXAMPLES=OFF \
			  -DBUILD_TESTS=OFF \
			  -DBUILD_PERF_TESTS=OFF \
			  -DWITH_CUDA=OFF \
			  -DWITH_GTK=OFF \
			  -DWITH_QT=OFF \
			  -DWITH_JPEG=ON \
			  -DWITH_PNG=ON \
			  -DWITH_TIFF=ON \
			  -DENABLE_NEON=ON \
			  .. && \
		make -j\$$(nproc) && \
		echo '🎯 Building application...' && \
		cd /workspace && \
		aarch64-linux-gnu-g++ onnx_infer.cpp -o onnx_infer_rpi \
				$(INCLUDE) \
				$(LIB) \
				$(OPENCV_3RDPARTY_LIBS) \
				$(ADDITIONAL_FLAGS) \
				$(RPATH) \
				$(STATIC_LINK_FLAGS) && \
		echo '✅ Build completed successfully!' && \
		echo '📁 Output: onnx_infer_rpi' \
	"



# Check Docker availability
check-docker:
	@echo "Checking Docker setup..."
	@if command -v docker >/dev/null 2>&1; then \
		echo "✅ Docker is installed"; \
		if docker info >/dev/null 2>&1; then \
			echo "✅ Docker daemon is running"; \
			echo "✅ Ready for cross-compilation!"; \
		else \
			echo "❌ Docker daemon is not running"; \
			echo "💡 Start with: sudo systemctl start docker"; \
			echo "💡 Enable on boot: sudo systemctl enable docker"; \
			exit 1; \
		fi; \
	else \
		echo "❌ Docker is not installed"; \
		echo "💡 Install with: sudo apt update && sudo apt install docker.io"; \
		echo "💡 Add user to docker group: sudo usermod -aG docker $$USER"; \
		echo "💡 Then logout and login again"; \
		exit 1; \
	fi

# Show OpenCV build status and library type
opencv-status:
	@echo "OpenCV Build Status:"
	@echo "==================="
	@echo "Build directory: $(OPENCV_BUILD_DIR)"
	@if [ -d "$(OPENCV_BUILD_DIR)" ]; then \
		echo "✓ Build directory exists"; \
		if [ -f "$(OPENCV_LIB_DIR)/libopencv_core.a" ]; then \
			echo "✓ Static libraries found (.a files)"; \
			echo "  - Will use static linking with all dependencies"; \
		elif [ -f "$(OPENCV_LIB_DIR)/libopencv_core.so" ]; then \
			echo "✓ Shared libraries found (.so files)"; \
			echo "  - Will use dynamic linking"; \
		else \
			echo "✗ No OpenCV libraries found"; \
		fi; \
		echo "Library directory: $(OPENCV_LIB_DIR)"; \
		echo "Available libraries:"; \
		ls -la $(OPENCV_LIB_DIR)/libopencv_* 2>/dev/null || echo "  No OpenCV libraries found"; \
	else \
		echo "✗ Build directory does not exist"; \
		echo "  Run 'make rpi-docker' to build OpenCV and application"; \
	fi

clean:
	rm -f onnx_infer_linux onnx_infer_mac onnx_infer_rpi

clean-opencv:
	rm -rf opencv/build

rebuild: clean all

# Help target
help:
	@echo "🔨 ONNX Inference Build System (Docker-focused)"
	@echo "=============================================="
	@echo ""
	@echo "📋 Main Targets:"
	@echo "  rpi-docker         🐳 Cross-compile for Raspberry Pi in Docker (RECOMMENDED)"
	@echo "  rpi-native         🏠 Build natively on Raspberry Pi"
	@echo "  all                💻 Build for current platform (x86_64/local dev)"
	@echo ""
	@echo "🔧 Development Targets:"
	@echo "  build-opencv-native  📚 Build OpenCV for local development"
	@echo "  opencv-status        📊 Show OpenCV build status and library type"
	@echo "  check-docker         🐳 Verify Docker installation and setup"
	@echo "  test                 🧪 Test the built executable"
	@echo "  clean                🧹 Remove built executables"
	@echo "  clean-opencv         🗑️  Remove OpenCV build directory"
	@echo ""
	@echo "🎯 Recommended Workflow:"
	@echo "  1. make check-docker              # Verify Docker is ready"
	@echo "  2. make rpi-docker                # Build for Raspberry Pi"
	@echo "  3. Copy onnx_infer_rpi to your Pi # Deploy the binary"
	@echo ""
	@echo "✨ Why Docker-only cross-compilation?"
	@echo "  ✅ Consistent build environment (Debian 11, glibc 2.31)"
	@echo "  ✅ No complex local ARM64 library setup required"
	@echo "  ✅ Handles all dependencies automatically"
	@echo "  ✅ Avoids glibc version conflicts"
	@echo "  ✅ Reproducible builds across different host systems"
	@echo ""
	@echo "📋 Prerequisites:"
	@echo "  • Docker installed and running"
	@echo "  • ONNX Runtime for ARM64: ./onnxruntime-linux-aarch64-1.23.1/"
	@echo "  • OpenCV source code: ./opencv/"
	@echo ""
	@echo "🚨 Need local cross-compilation? Consider Docker first - it's more reliable!"

.PHONY: all check-opencv build-opencv-native opencv-status check-docker test clean clean-opencv rebuild rpi-native rpi-docker help