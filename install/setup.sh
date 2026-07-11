#!/bin/bash
set -e

# Get environment name
if [ -z "$1" ]; then
    read -p "Enter environment name (default: arm_nbv): " ENV_NAME
    ENV_NAME=${ENV_NAME:-arm_nbv}
else
    ENV_NAME=$1
fi

# Create environment
echo "Step 1/4: Creating and activating environment $ENV_NAME with Python 3.10..."
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

echo ""
# Detect CUDA: check for nvidia-smi binary OR /dev/nvidia0 device node.
# nvidia-smi may fail (exit 9) if the driver isn't communicating yet, so use its mere presence.
if command -v nvidia-smi &> /dev/null || [ -e /dev/nvidia0 ]; then
    # Install PyTorch from pip with the official cu128 index.
    # nvblox_torch is compiled against official pip pytorch wheels (cu128 ABI)
    # and can fail with conda installs ("undefined symbol: iJIT_NotifyEvent").
    echo ""
    echo "Step 2/4: CUDA detected - installing PyTorch cu128 (pip) and nvblox_torch (pip)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

    # nvblox (NVIDIA 3D reconstruction, direct wheel install)
    pip install https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.9/nvblox_torch-0.0.9+cu12ubuntu24-py3-none-linux_x86_64.whl
else
    echo ""
    echo "Step 2/4: No CUDA detected - Installing PyTorch (conda, cpu)..."
    conda install pytorch torchvision cpuonly -c pytorch -y
fi

# Install remaining packages from yml
echo ""
echo "Step 3/4: Installing remaining packages from environment.yml..."
conda env update --name "$ENV_NAME" --file install/environment.yml --verbose

# Install repository in editable mode
echo ""
echo "Step 4/4: Installing repository packages in editable mode..."
pip install -e external/robosuite
pip install -e .

echo ""
echo "Installation and activation of '$ENV_NAME' completed!"
