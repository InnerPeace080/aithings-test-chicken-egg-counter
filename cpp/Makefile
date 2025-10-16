SRC = onnx_infer.cpp
COMPILER = g++

INCLUDE = -I$(ONNX_RUNTIME_DIR)/include
LIB = -L$(ONNX_RUNTIME_DIR)/lib -lonnxruntime

# check platform and set flags accordingly
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	TARGET = onnx_infer_linux
	ONNX_RUNTIME_DIR = ./onnxruntime-linux-x64-1.23.1
	RPATH = -Wl,-rpath,'$$ORIGIN/$(ONNX_RUNTIME_DIR)/lib'

	ADDITIONAL_FLAGS = -std=c++14
else ifeq ($(UNAME_S),Darwin)		
	TARGET = onnx_infer_mac
	ONNX_RUNTIME_DIR = ./onnxruntime-osx-arm64-1.23.1
	RPATH = -Wl,-rpath,@executable_path/$(ONNX_RUNTIME_DIR)/lib

	ADDITIONAL_FLAGS = -std=c++14
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $^ -o $@ $(INCLUDE) $(LIB) $(ADDITIONAL_FLAGS) $(RPATH)

clean:
	rm -f $(TARGET)

rebuild: clean all