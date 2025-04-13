#!/bin/bash

# Exit on error
set -e

# Set up logging
LOG_FILE="/workspace/logs/entrypoint.log"
mkdir -p /workspace/logs
touch $LOG_FILE

echo "--- Log Start: $(date -u) ---" | tee -a $LOG_FILE

# Function to log information
log_info() {
  echo "INFO: $1" | tee -a $LOG_FILE
}

# Function to log warnings
log_warning() {
  echo "WARNING: $1" | tee -a $LOG_FILE
}

# Function to log errors
log_error() {
  echo "ERROR: $1" | tee -a $LOG_FILE
}

# Set up configuration variables
log_info "Setting up configuration variables..."
WORKSPACE_DIR="/workspace"
SHARED_DIR="/workspace/shared"
A1111_DIR="/workspace/stable-diffusion-webui"
RUN_MODE="a1111"
USER_A1111_ARGS="${A1111_ARGS:---xformers --api --port 17860}"
HF_TOKEN_SET="${HF_TOKEN:+set}"
CIVITAI_TOKEN_SET="${CIVITAI_TOKEN:+set}"

log_info "Workspace Directory: $WORKSPACE_DIR"
log_info "Shared Directory: $SHARED_DIR"
log_info "A1111 Directory: $A1111_DIR"
log_info "Run Mode: $RUN_MODE"
log_info "User Provided A1111_ARGS: $USER_A1111_ARGS"
log_info "HF_TOKEN is ${HF_TOKEN_SET:+set}"
log_info "CIVITAI_TOKEN is ${CIVITAI_TOKEN_SET:+set}"

# Create directories
log_info "Creating directory structure..."
mkdir -p $WORKSPACE_DIR $SHARED_DIR $A1111_DIR/models/Stable-diffusion $A1111_DIR/models/Lora $A1111_DIR/models/ControlNet

# Store tokens
if [ -n "$CIVITAI_TOKEN" ]; then
  log_info "Storing Civitai token..."
  mkdir -p $HOME/.cache/civitai
  echo "$CIVITAI_TOKEN" > $HOME/.cache/civitai/civitai.token
  log_info "Civitai token stored."
fi

if [ -n "$HF_TOKEN" ]; then
  log_info "Storing Hugging Face token..."
  mkdir -p $HOME/.huggingface
  echo "$HF_TOKEN" > $HOME/.huggingface/token
  log_info "Hugging Face token stored."
fi

log_info "Effective HF_TOKEN is ${HF_TOKEN_SET:+set}"
log_info "Effective CIVITAI_TOKEN is ${CIVITAI_TOKEN_SET:+set}"

# Install system packages
log_info "Installing essential system packages..."
apt-get update && apt-get install -y --no-install-recommends unzip wget libgl1-mesa-glx curl git libglib2.0-0 aria2

# Fix PyTorch environment
fix_pytorch_environment() {
  log_info "Diagnosing PyTorch environment..."
  
  # Find Python versions and paths
  which python3
  python3 --version
  
  # Check if PyTorch is installed in any Python environment
  pip list | grep torch || echo "PyTorch not found in pip list"
  
  # Try to find torch installation
  find / -name "torch" -type d 2>/dev/null | grep -v "__pycache__" || echo "No torch directories found"
  
  # Install PyTorch if not found
  log_info "Installing PyTorch to ensure availability..."
  
  # Use environment variables if set, otherwise use defaults
  PYTORCH_VERSION=${PYTORCH_VERSION:-2.5.1}
  PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
  
  pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url $PYTORCH_INDEX_URL
  
  # Update PYTHONPATH to include common locations
  export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages
  
  # Pin critical dependencies
  pip install insightface==0.7.3 onnxruntime-gpu==1.15.1
  
  # Validate installation
  python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
  
  # Return status but don't fail if PyTorch check fails
  if [ $? -ne 0 ]; then
    log_warning "PyTorch validation failed but continuing..."
    return 1
  else
    log_info "PyTorch successfully found/installed with CUDA support"
    return 0
  fi
}

# Run the PyTorch fix
fix_pytorch_environment

# Set CUDA environment variables for performance
log_info "Setting CUDA environment variables for performance..."
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# Validate CUDA and PyTorch
log_info "Validating CUDA and PyTorch from base image using python3..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Using device: {device}')" > /workspace/logs/cuda_init.log 2>&1

if [ $? -ne 0 ]; then
  log_error "CUDA validation failed using python3. Check /workspace/logs/cuda_init.log and /workspace/logs/entrypoint.log for details."
  # Continue despite error
fi

# Clone or update the A1111 repository
if [ ! -d "$A1111_DIR/.git" ]; then
  log_info "Cloning Stable Diffusion WebUI repository..."
  git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git $A1111_DIR
  cd $A1111_DIR
  git checkout v1.10.1
else
  log_info "Updating existing Stable Diffusion WebUI repository..."
  cd $A1111_DIR
  git pull
  git checkout v1.10.1
fi

# Define extensions to install
EXTENSIONS=(
  "https://github.com/Mikubill/sd-webui-controlnet.git"
  "https://github.com/fkunn1326/openpose-editor.git"
  "https://github.com/camenduru/stable-diffusion-webui-huggingface.git"
  "https://github.com/camenduru/stable-diffusion-webui-tunnels.git"
  "https://github.com/etherealxx/batchlinks-webui.git"
  "https://github.com/camenduru/stable-diffusion-webui-catppuccin.git"
  "https://github.com/KohakuBlueleaf/a1111-sd-webui-locon.git"
  "https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg.git"
  "https://github.com/ashen-sensored/stable-diffusion-webui-two-shot.git"
  "https://github.com/thomasasfk/sd-webui-aspect-ratio-helper.git"
  "https://github.com/tjm35/asymmetric-tiling-sd-webui.git"
  "https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111.git"
  "https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git"
  "https://github.com/kohya-ss/sd-webui-additional-networks.git"
  "https://github.com/AlUlkesh/stable-diffusion-webui-images-browser.git"
  "https://github.com/continue-revolution/sd-webui-segment-anything.git"
  "https://github.com/civitai/sd_civitai_extension.git"
  "https://github.com/Gourieff/sd-webui-reactor.git"
  "https://github.com/fkunn1326/openpose-editor.git"
  "https://github.com/hnmr293/posex.git"
  "https://github.com/nonnonstop/sd-webui-3d-open-pose-editor.git"
  "https://github.com/v8hid/infinite-image-browsing.git"
  "https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor.git"
  "https://github.com/toriato/stable-diffusion-webui-wd14-tagger.git"
  "https://github.com/adieyal/sd-dynamic-prompts.git"
  "https://github.com/ControlNet-org/ControlNet-v1-Unified.git"
)

# Install extensions
log_info "Installing extensions..."
cd $A1111_DIR/extensions
for ext in "${EXTENSIONS[@]}"; do
  ext_name=$(basename $ext .git)
  if [ ! -d "$ext_name" ]; then
    log_info "Cloning extension: $ext_name"
    git clone $ext || log_warning "Failed to clone $ext, continuing..."
  else
    log_info "Extension $ext_name already exists, updating..."
    cd $ext_name
    git pull || log_warning "Failed to update $ext_name, continuing..."
    cd ..
  fi
done

# Define targeted downloads for specific paths
declare -a TARGETED_URLS=(
  "https://huggingface.co/sam-hq-team/sam-hq-vit-h/resolve/main/sam_hq_vit_h.pth"
)

declare -a TARGETED_PATHS=(
  "$A1111_DIR/extensions/sd-webui-segment-anything/models/sam_hq_vit_h.pth"
)

# Function for targeted downloads
provisioning_download_targeted() {
  local url=$1
  local path=$2
  local dir=$(dirname "$path")
  
  mkdir -p "$dir"
  
  if [ ! -f "$path" ]; then
    log_info "Downloading $(basename "$path")..."
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "$url" -d "$dir" -o "$(basename "$path")" || {
      log_warning "Failed to download $(basename "$path") using aria2c, trying wget..."
      wget -q "$url" -O "$path" || log_warning "Failed to download $(basename "$path") using wget as well, continuing..."
    }
  else
    log_info "$(basename "$path") already exists, skipping download."
  fi
}

# Perform targeted downloads
log_info "Performing targeted downloads..."
for i in "${!TARGETED_URLS[@]}"; do
  provisioning_download_targeted "${TARGETED_URLS[$i]}" "${TARGETED_PATHS[$i]}"
done

# Define models to download
LORA_MODELS=(
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
  "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
)

CONTROLNET_MODELS=(
  "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_mid.safetensors"
  "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_depth_mid.safetensors"
)

# Function to download models
download_model() {
  local url=$1
  local dir=$2
  local filename=$(basename "$url")
  
  if [ ! -f "$dir/$filename" ]; then
    log_info "Downloading $filename..."
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "$url" -d "$dir" -o "$filename" || {
      log_warning "Failed to download $filename using aria2c, trying wget..."
      wget -q "$url" -O "$dir/$filename" || log_warning "Failed to download $filename using wget as well, continuing..."
    }
  else
    log_info "$filename already exists, skipping download."
  fi
}

# Download models
log_info "Downloading Lora models..."
for model in "${LORA_MODELS[@]}"; do
  download_model "$model" "$A1111_DIR/models/Stable-diffusion"
done

log_info "Downloading ControlNet models..."
for model in "${CONTROLNET_MODELS[@]}"; do
  download_model "$model" "$A1111_DIR/models/ControlNet"
done

# Enforce required arguments
ENFORCED_ARGS="--xformers --api --no-half-vae"
FINAL_ARGS="$USER_A1111_ARGS"

# Check if each enforced arg is already present in USER_A1111_ARGS
for arg in $ENFORCED_ARGS; do
  if [[ ! "$FINAL_ARGS" =~ $arg ]]; then
    log_info "Adding enforced argument: $arg"
    FINAL_ARGS="$FINAL_ARGS $arg"
  fi
done

# Start the WebUI
log_info "Starting Stable Diffusion WebUI with arguments: $FINAL_ARGS"
cd $A1111_DIR
python launch.py $FINAL_ARGS
