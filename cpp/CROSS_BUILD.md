# Cross-Building for Raspberry Pi

This document provides detailed instructions for cross-compiling the chicken-egg counter application for Raspberry Pi platforms.

**Note**: This guide assumes you are running commands from within the `cpp/` directory of the project.

## Overview

Cross-compilation allows you to build executables for Raspberry Pi on your development machine (macOS/Linux x86_64) without needing to compile directly on the target device. This is faster and more convenient for development workflows.

**Important for macOS Users**: Cross-compilation on macOS is more complex than on Linux because macOS doesn't have readily available Linux cross-compilation toolchains. The recommended approach for macOS users is to use Docker (see Docker section below) or build natively on a Raspberry Pi.

## Prerequisites

### 1. Cross-Compilation Toolchain

Install the appropriate GCC cross-compiler for your target Raspberry Pi:

#### For 64-bit ARM (Raspberry Pi 4/5 with 64-bit OS)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# macOS - Option 1: Using Docker (Recommended)
# Cross-compilation on macOS is complex. Use Docker for consistent results.
# See Docker section below for setup instructions.

# macOS - Option 2: Manual toolchain (Advanced users)
# You can build a cross-compilation toolchain using crosstool-ng:
# brew install crosstool-ng
# ct-ng aarch64-unknown-linux-gnu
# ct-ng build
```

#### For 32-bit ARM (Raspberry Pi 3/4 with 32-bit OS)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

# macOS - Option 1: Using Docker (Recommended)
# Cross-compilation on macOS is complex. Use Docker for consistent results.
# See Docker section below for setup instructions.

# macOS - Option 2: Manual toolchain (Advanced users)
# You can build a cross-compilation toolchain using crosstool-ng:
# brew install crosstool-ng
# ct-ng arm-unknown-linux-gnueabihf
# ct-ng build
```

### 2. ONNX Runtime for ARM

Download the ARM version of ONNX Runtime from the [official releases](https://github.com/microsoft/onnxruntime/releases):

#### For 64-bit ARM

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-aarch64-1.23.1.tgz
tar -xzf onnxruntime-linux-aarch64-1.23.1.tgz
```

#### For 32-bit ARM

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-armhf-1.23.1.tgz
tar -xzf onnxruntime-linux-armhf-1.23.1.tgz
```

## macOS Users: Recommended Approach

Since cross-compilation toolchains for Linux targets are not readily available on macOS, we recommend one of these approaches:

### Option 1: Docker (Recommended)

Use Docker to create a Linux environment with cross-compilation tools:

```bash
# Create a Dockerfile in the cpp directory
cat > Dockerfile << 'EOF'
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    cmake \
    make \
    wget \
    build-essential

WORKDIR /build
EOF

# Build the Docker image
docker build -t rpi-cross-build .

# Run cross-compilation in container
docker run --rm -v $(pwd):/build -w /build rpi-cross-build make rpi-cross CROSS_COMPILE=aarch64-linux-gnu-
```

### Option 2: Native Compilation on Raspberry Pi

For simpler projects, you can compile directly on the Raspberry Pi:

```bash
# Copy source files to Raspberry Pi
scp -r . pi@your-pi-ip:~/chicken-egg-counter/

# SSH to Raspberry Pi and build
ssh pi@your-pi-ip
cd ~/chicken-egg-counter/cpp
make rpi
```

## Building Process

### Step 1: Build OpenCV (Cross-compiled)

First, cross-compile OpenCV for the target platform:

**Note**: The following commands work on Linux or in Docker containers. For macOS, use the Docker approach described above.

**For 64-bit ARM:**

```bash
make build-opencv-rpi CROSS_COMPILE=aarch64-linux-gnu-
```

**For 32-bit ARM:**

```bash
make build-opencv-rpi CROSS_COMPILE=arm-linux-gnueabihf-
```

### Step 2: Cross-compile the Application

**For 64-bit ARM:**

```bash
make rpi-cross CROSS_COMPILE=aarch64-linux-gnu-
```

**For 32-bit ARM:**

```bash
make rpi-cross CROSS_COMPILE=arm-linux-gnueabihf-
```

### Step 3: Verify the Build

Check that the executable was built correctly:

```bash
make test TARGET_PLATFORM=RaspberryPi CROSS_COMPILE=aarch64-linux-gnu-
```

## Available Make Targets

### Cross-compilation Targets

- `rpi-cross` - Cross-compile the main application for Raspberry Pi
- `build-opencv-rpi` - Cross-compile OpenCV for Raspberry Pi
- `help` - Display all available targets and usage examples

### Usage Examples

```bash
# Cross-compile everything for 64-bit ARM Raspberry Pi
make build-opencv-rpi CROSS_COMPILE=aarch64-linux-gnu-
make rpi-cross CROSS_COMPILE=aarch64-linux-gnu-

# Cross-compile for 32-bit ARM Raspberry Pi
make build-opencv-rpi CROSS_COMPILE=arm-linux-gnueabihf-
make rpi-cross CROSS_COMPILE=arm-linux-gnueabihf-

# Clean and rebuild
make clean
make rpi-cross CROSS_COMPILE=aarch64-linux-gnu-
```

## Deployment to Raspberry Pi

### Method 1: SCP Transfer

Copy the built files to your Raspberry Pi:

```bash
# Copy the executable
scp onnx_infer_rpi pi@your-pi-ip:~/

# Copy the ONNX Runtime libraries
scp -r onnxruntime-linux-aarch64-1.23.1 pi@your-pi-ip:~/

# Copy OpenCV libraries
scp -r opencv/build/lib pi@your-pi-ip:~/opencv-lib/

# Copy model files (from project root)
scp ../models/*.onnx pi@your-pi-ip:~/models/
```

### Method 2: Build Script

Create a deployment script:

```bash
#!/bin/bash
# deploy.sh

PI_IP="your-pi-ip"
PI_USER="pi"
APP_DIR="chicken-egg-counter"

# Create directory on Pi
ssh ${PI_USER}@${PI_IP} "mkdir -p ~/${APP_DIR}/{models,libs}"

# Copy executable
scp onnx_infer_rpi ${PI_USER}@${PI_IP}:~/${APP_DIR}/

# Copy libraries
scp -r onnxruntime-linux-aarch64-1.23.1/lib/* ${PI_USER}@${PI_IP}:~/${APP_DIR}/libs/
scp -r opencv/build/lib/* ${PI_USER}@${PI_IP}:~/${APP_DIR}/libs/

# Copy models (from project root)
scp ../models/*.onnx ${PI_USER}@${PI_IP}:~/${APP_DIR}/models/

echo "Deployment complete!"
```

## Running on Raspberry Pi

Once deployed, run the application on your Raspberry Pi:

```bash
# SSH to your Raspberry Pi
ssh pi@your-pi-ip

# Navigate to the application directory
cd ~/chicken-egg-counter

# Set library path and run
export LD_LIBRARY_PATH=./libs:$LD_LIBRARY_PATH
./onnx_infer_rpi models/yolo_chicken_egg_infer.onnx input_image.jpg
```

## Troubleshooting

### Common Issues

1. **Missing cross-compiler**: Ensure the cross-compilation toolchain is properly installed
2. **ONNX Runtime not found**: Verify the ARM version of ONNX Runtime is downloaded and extracted
3. **Library path issues**: Ensure `LD_LIBRARY_PATH` includes the libraries directory on the Pi
4. **Architecture mismatch**: Make sure you're using the correct toolchain (32-bit vs 64-bit) for your Raspberry Pi OS

### Debug Commands

```bash
# Check executable architecture
file onnx_infer_rpi

# Check library dependencies
aarch64-linux-gnu-objdump -p onnx_infer_rpi | grep NEEDED

# Verify cross-compiler installation
aarch64-linux-gnu-gcc --version
```

## Performance Notes

- Cross-compiled applications should have similar performance to natively compiled ones
- The ARM-specific compiler flags (`-march=armv7-a -mfpu=neon-vfpv4`) optimize for Raspberry Pi hardware
- For best performance on Raspberry Pi 4/5, use the 64-bit toolchain and OS

## Docker-based Cross-compilation (Recommended for macOS)

For a more isolated and consistent build environment, especially on macOS, use Docker:

### Complete Docker Setup

Create a comprehensive Dockerfile:

```dockerfile
FROM ubuntu:20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    cmake \
    make \
    wget \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/cpp
COPY . .

# Download ONNX Runtime if not present
RUN if [ ! -d "onnxruntime-linux-aarch64-1.23.1" ]; then \
        wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-aarch64-1.23.1.tgz && \
        tar -xzf onnxruntime-linux-aarch64-1.23.1.tgz && \
        rm onnxruntime-linux-aarch64-1.23.1.tgz; \
    fi
```

### Build Commands

```bash
# Build Docker image (run from project root)
docker build -f cpp/Dockerfile -t chicken-egg-cross .

# Cross-compile for 64-bit ARM
docker run --rm -v $(pwd)/cpp:/build/cpp -w /build/cpp chicken-egg-cross make rpi-cross CROSS_COMPILE=aarch64-linux-gnu-

# Cross-compile for 32-bit ARM
docker run --rm -v $(pwd)/cpp:/build/cpp -w /build/cpp chicken-egg-cross make rpi-cross CROSS_COMPILE=arm-linux-gnueabihf-

# Copy built executable to host
docker run --rm -v $(pwd)/output:/output -v $(pwd)/cpp:/build/cpp chicken-egg-cross cp /build/cpp/onnx_infer_rpi /output/
```

### One-liner for macOS Users

```bash
# Quick build for 64-bit ARM Raspberry Pi
docker run --rm -v $(pwd)/cpp:/build -w /build ubuntu:20.04 bash -c "
apt-get update && apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu make cmake wget build-essential &&
if [ ! -d 'onnxruntime-linux-aarch64-1.23.1' ]; then
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-aarch64-1.23.1.tgz &&
    tar -xzf onnxruntime-linux-aarch64-1.23.1.tgz
fi &&
make rpi-cross CROSS_COMPILE=aarch64-linux-gnu-"
```
