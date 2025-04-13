#!/bin/bash

# Exit on error
set -e

# Set up logging
LOG_FILE="/workspace/logs/entrypoint.log"
mkdir -p /workspace/logs
touch $LOG_FILE

# Function to append to log and print to stdout
log_message() {
  echo "$1" | tee -a $LOG_FILE
}

# Function to log information
log_info() {
  log_message "INFO: $1"
}

# Function to log warnings
log_warning() {
  log_message "WARNING: $1"
}

# Function to log errors
log_error() {
  log_message "ERROR: $1"
}

echo "--- Log Start: $(date -u) ---" | tee -a $LOG_FILE

# Set up configuration variables
log_info "Setting up configuration variables..."
WORKSPACE_DIR="/workspace"
SHARED_DIR="/workspace/shared"
A1111_DIR="/workspace/stable-diffusion-webui"
RUN_MODE="a1111" # Assuming this might be used later, keeping it.
USER_A1111_ARGS="${A1111_ARGS:---xformers --api --port 17860}"
HF_TOKEN_SET="${HF_TOKEN:+set}"
CIVITAI_TOKEN_SET="${CIVITAI_TOKEN:+set}"

log_info "Workspace Directory: $WORKSPACE_DIR"
log_info "Shared Directory: $SHARED_DIR"
log_info "A1111 Directory: $A1111_DIR"
log_info "Run Mode: $RUN_MODE"
log_info "User Provided A1111_ARGS: $USER_A1111_ARGS"
log_info "HF_TOKEN is ${HF_TOKEN_SET:-not set}"
log_info "CIVITAI_TOKEN is ${CIVITAI_TOKEN_SET:-not set}"

# Create directories
log_info "Creating directory structure..."
mkdir -p $WORKSPACE_DIR $SHARED_DIR $A1111_DIR/models/Stable-diffusion $A1111_DIR/models/Lora $A1111_DIR/models/ControlNet $A1111_DIR/models/VAE $A1111_DIR/models/hypernetworks

# Store tokens
if [ -n "$CIVITAI_TOKEN" ]; then
  log_info "Storing Civitai token..."
  mkdir -p "$(dirname "$HOME/.cache/civitai/civitai.token")"
  echo "$CIVITAI_TOKEN" > "$HOME/.cache/civitai/civitai.token"
  log_info "Civitai token stored."
fi

if [ -n "$HF_TOKEN" ]; then
  log_info "Storing Hugging Face token..."
  mkdir -p "$(dirname "$HOME/.huggingface/token")"
  echo "$HF_TOKEN" > "$HOME/.huggingface/token"
  log_info "Hugging Face token stored."
fi

log_info "Effective HF_TOKEN is ${HF_TOKEN_SET:-not set}"
log_info "Effective CIVITAI_TOKEN is ${CIVITAI_TOKEN_SET:-not set}"

# Install system packages
log_info "Installing essential system packages..."
# Added timeout to prevent indefinite hangs
apt-get update && apt-get install -y --no-install-recommends timeout unzip wget libgl1-mesa-glx curl git libglib2.0-0 aria2

# Fix PyTorch environment
fix_pytorch_environment() {
  log_info "Diagnosing PyTorch environment..."

  # Find Python versions and paths
  log_info "Python path: $(which python3 || echo 'python3 not found')"
  log_info "Python version: $(python3 --version || echo 'python3 failed')"

  # Check if PyTorch is installed in any Python environment
  log_info "Checking pip list for torch..."
  pip list | grep torch || log_info "PyTorch not found in pip list"

  # Try to find torch installation (might be slow, keep commented unless debugging)
  # log_info "Searching for torch directories..."
  # find / -name "torch" -type d 2>/dev/null | grep -v "__pycache__" || log_info "No torch directories found"

  # Install PyTorch if not found or validation fails
  log_info "Attempting to install/validate PyTorch..."

  # Use environment variables if set, otherwise use defaults
  PYTORCH_VERSION=${PYTORCH_VERSION:-2.1.0} # Adjusted to common stable version
  PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

  # Attempt installation - use timeout to prevent hangs
  log_info "Running: timeout 600 pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url $PYTORCH_INDEX_URL"
  if ! timeout 600 pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url $PYTORCH_INDEX_URL; then
      log_warning "Initial PyTorch install failed. Trying without version pinning..."
      if ! timeout 600 pip install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL; then
         log_error "PyTorch installation failed even without version pin. Check network and index URL."
         return 1 # Signal failure
      fi
  fi
  log_info "PyTorch install command finished."

  # Update PYTHONPATH just in case
  export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages
  log_info "PYTHONPATH: $PYTHONPATH"

  # Pin critical dependencies (adjust versions as needed for compatibility)
  log_info "Installing insightface and onnxruntime-gpu..."
  if ! timeout 300 pip install insightface==0.7.3 onnxruntime-gpu==1.15.1; then
      log_error "Failed to install insightface or onnxruntime-gpu."
      return 1 # Signal failure
  fi
  log_info "Insightface and ONNX Runtime install command finished."

  # Validate installation
  log_info "Validating PyTorch installation..."
  if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"; then
    log_info "PyTorch successfully validated with CUDA support"
    return 0
  else
    log_error "PyTorch validation failed. Check previous installation logs."
    return 1 # Signal failure
  fi
}

# Run the PyTorch fix - exit if it fails
log_info "Running PyTorch environment fix..."
if ! fix_pytorch_environment; then
    log_error "PyTorch setup failed. Exiting."
    exit 1
fi
log_info "PyTorch environment fix completed."


# Set CUDA environment variables for performance
log_info "Setting CUDA environment variables for performance..."
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# Validate CUDA and PyTorch again after setup
log_info "Validating CUDA and PyTorch from environment using python3..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Using device: {device}')" > /workspace/logs/cuda_check.log 2>&1

if [ $? -ne 0 ]; then
  log_error "PyTorch/CUDA validation failed using python3. Check /workspace/logs/cuda_check.log and /workspace/logs/entrypoint.log for details."
  exit 1 # Exit if validation fails after setup
fi
log_info "PyTorch/CUDA validation successful."


# Clone or update the A1111 repository
A1111_VERSION="v1.10.1" # Define version here
log_info "Setting up A1111 repository (target version: $A1111_VERSION)..."
if [ ! -d "$A1111_DIR/.git" ]; then
  log_info "Cloning Stable Diffusion WebUI repository..."
  git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git $A1111_DIR
  cd $A1111_DIR
  log_info "Checking out specific version: $A1111_VERSION"
  git checkout $A1111_VERSION
else
  log_info "Updating existing Stable Diffusion WebUI repository..."
  cd $A1111_DIR
  git fetch --all # Fetch all changes
  # Stash local changes if any, to prevent checkout conflicts
  if ! git stash push -m "entrypoint-stash"; then
     log_warning "git stash failed, potential local changes might cause issues."
  fi
  log_info "Checking out specific version: $A1111_VERSION"
  if ! git checkout $A1111_VERSION; then
     log_error "Failed to checkout A1111 version $A1111_VERSION. Check for conflicts or invalid tag."
     exit 1
  fi
  # Optionally pull LFS files if needed after checkout (usually handled by launch.py)
  # git lfs pull
fi
log_info "A1111 repository setup complete."

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
  # Removed duplicate openpose-editor
  "https://github.com/hnmr293/posex.git"
  "https://github.com/nonnonstop/sd-webui-3d-open-pose-editor.git"
  "https://github.com/v8hid/infinite-image-browsing.git"
  "https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor.git"
  "https://github.com/toriato/stable-diffusion-webui-wd14-tagger.git"
  "https://github.com/adieyal/sd-dynamic-prompts.git"
  "https://github.com/ControlNet-org/ControlNet-v1-Unified.git" # This might be better placed elsewhere or handled by sd-webui-controlnet
)

# Install extensions
log_info "Installing/Updating extensions..."
EXTENSIONS_DIR="$A1111_DIR/extensions"
mkdir -p "$EXTENSIONS_DIR"
cd "$EXTENSIONS_DIR"
for ext_url in "${EXTENSIONS[@]}"; do
  ext_name=$(basename "$ext_url" .git)
  ext_path="$EXTENSIONS_DIR/$ext_name"
  log_info "Processing extension: $ext_name from $ext_url"
  if [ ! -d "$ext_path/.git" ]; then
    log_info "Cloning extension: $ext_name"
    # Use timeout for clone
    if ! timeout 300 git clone "$ext_url"; then
        log_warning "Failed to clone $ext_name (timed out or error). Continuing..."
        # Clean up potentially broken clone directory
        rm -rf "$ext_path"
    fi
  else
    log_info "Extension $ext_name already exists, updating..."
    cd "$ext_path"
    # Fetch updates and reset to remote HEAD (overwrite local changes)
    if ! timeout 180 git fetch --all; then
        log_warning "git fetch failed for $ext_name (timed out or error). Skipping update."
        cd ..
        continue # Skip to next extension
    fi
    # Get default branch name
    DEFAULT_BRANCH=$(git remote show origin | grep 'HEAD branch' | cut -d' ' -f5)
    if [ -z "$DEFAULT_BRANCH" ]; then
        log_warning "Could not determine default branch for $ext_name. Attempting reset to origin/HEAD."
        DEFAULT_BRANCH="HEAD" # Fallback
    fi
    log_info "Resetting $ext_name to origin/$DEFAULT_BRANCH"
    if ! timeout 60 git reset --hard "origin/$DEFAULT_BRANCH"; then
        log_warning "git reset failed for $ext_name (timed out or error). Skipping update."
    fi
    # Optionally pull LFS files if the extension uses them
    # timeout 180 git lfs pull || log_warning "git lfs pull failed for $ext_name"
    cd ..
  fi
done
log_info "Extension processing complete."

# --- Robust Download Functions ---

# Function for targeted downloads (like specific model files for extensions)
# Returns 0 on success, 1 on failure.
provisioning_download_targeted() {
  local url="$1"
  local path="$2"
  local filename=$(basename "$path")
  local dir=$(dirname "$path")

  log_info "Starting targeted download for $filename..."
  log_info "URL: $url"
  log_info "Destination: $path"

  mkdir -p "$dir"

  if [ -f "$path" ]; then
    log_info "$filename already exists at $path, skipping download."
    return 0
  fi

  log_info "Attempting download using aria2c..."
  # Use timeout for download command
  if timeout 1800 aria2c --console-log-level=warn --summary-interval=10 -c -x 16 -s 16 -k 1M "$url" -d "$dir" -o "$filename"; then
    log_info "$filename downloaded successfully to $path using aria2c."
    return 0
  else
    local exit_code=$?
    log_warning "aria2c failed for $filename (exit code $exit_code or timed out). Removing partial file (if any) and trying wget..."
    rm -f "$path" # Ensure partial file is removed

    log_info "Attempting download using wget..."
    # Use timeout for download command
    if timeout 1800 wget -q --show-progress -O "$path" "$url"; then
      log_info "$filename downloaded successfully to $path using wget."
      return 0
    else
      local exit_code=$?
      log_error "wget also failed for $filename (exit code $exit_code or timed out). Download unsuccessful."
      rm -f "$path" # Ensure partial file is removed
      return 1 # Explicitly signal failure
    fi
  fi
}

# Function to download models into specified directories
# Returns 0 on success, 1 on failure.
download_model() {
  local url="$1"
  local dir="$2"
  local filename=$(basename "$url")
  local filepath="$dir/$filename"

  log_info "Starting model download for $filename..."
  log_info "URL: $url"
  log_info "Destination Directory: $dir"

  mkdir -p "$dir"

  if [ -f "$filepath" ]; then
    # Optional: Add file size check here if needed
    log_info "$filename already exists in $dir, skipping download."
    return 0
  fi

  log_info "Attempting download using aria2c..."
   # Use timeout for download command
  if timeout 3600 aria2c --console-log-level=warn --summary-interval=10 -c -x 16 -s 16 -k 1M "$url" -d "$dir" -o "$filename"; then
    log_info "$filename downloaded successfully to $filepath using aria2c."
    return 0
  else
    local exit_code=$?
    log_warning "aria2c failed for $filename (exit code $exit_code or timed out). Removing partial file (if any) and trying wget..."
    rm -f "$filepath" # Ensure partial file is removed

    log_info "Attempting download using wget..."
     # Use timeout for download command
    if timeout 3600 wget -q --show-progress -O "$filepath" "$url"; then
      log_info "$filename downloaded successfully to $filepath using wget."
      return 0
    else
      local exit_code=$?
      log_error "wget also failed for $filename (exit code $exit_code or timed out). Download unsuccessful."
      rm -f "$filepath" # Ensure partial file is removed
      return 1 # Explicitly signal failure
    fi
  fi
}

# --- Model Downloads ---

# Test connectivity first
log_info "Testing connectivity to Hugging Face..."
if timeout 60 curl -s --head https://huggingface.co > /dev/null; then
  log_info "Connectivity to Hugging Face OK."
else
  log_warning "Could not reach Hugging Face (timed out or error). Downloads may fail."
  # Decide if you want to exit here if HF is unreachable
  # exit 1
fi

# Define targeted downloads (example for SAM)
TARGETED_URLS=(
  "https://huggingface.co/sam-hq-team/sam-hq-vit-h/resolve/main/sam_hq_vit_h.pth"
)
TARGETED_PATHS=(
  "$A1111_DIR/extensions/sd-webui-segment-anything/models/sam_hq_vit_h.pth"
)

log_info "Performing targeted downloads..."
for i in "${!TARGETED_URLS[@]}"; do
  # Make this download non-critical (script continues even if it fails)
  if ! provisioning_download_targeted "${TARGETED_URLS[$i]}" "${TARGETED_PATHS[$i]}"; then
      log_warning "Optional download failed for $(basename ${TARGETED_PATHS[$i]}). Continuing..."
  fi
done


# Define models to download
# Note: Changed LORA_MODELS to SD_MODELS as these are base checkpoints
SD_MODELS=(
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
  "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
)

CONTROLNET_MODELS=(
  "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_mid.safetensors"
  "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_depth_mid.safetensors"
  # Add more ControlNet models here if needed
)

# Download Stable Diffusion models
log_info "Downloading Stable Diffusion models..."
SD_MODEL_DIR="$A1111_DIR/models/Stable-diffusion"
# Make the SDXL Base model download CRITICAL
log_info "Downloading SDXL Base (Critical)..."
if ! download_model "${SD_MODELS[0]}" "$SD_MODEL_DIR"; then
  log_error "Failed to download CRITICAL model $(basename ${SD_MODELS[0]}). Aborting."
  exit 1
fi
# Download other SD models (make them non-critical or critical as needed)
log_info "Downloading SDXL Refiner (Non-Critical)..."
if ! download_model "${SD_MODELS[1]}" "$SD_MODEL_DIR"; then
  log_warning "Failed to download non-critical model $(basename ${SD_MODELS[1]}). Continuing..."
fi

# Download ControlNet models
log_info "Downloading ControlNet models..."
CONTROLNET_MODEL_DIR="$A1111_DIR/models/ControlNet"
# Make the first ControlNet model CRITICAL (adjust as needed)
log_info "Downloading ControlNet Canny (Critical)..."
if ! download_model "${CONTROLNET_MODELS[0]}" "$CONTROLNET_MODEL_DIR"; then
  log_error "Failed to download CRITICAL model $(basename ${CONTROLNET_MODELS[0]}). Aborting."
  exit 1
fi
# Download other ControlNet models (make them non-critical or critical as needed)
log_info "Downloading ControlNet Depth (Non-Critical)..."
if ! download_model "${CONTROLNET_MODELS[1]}" "$CONTROLNET_MODEL_DIR"; then
  log_warning "Failed to download non-critical model $(basename ${CONTROLNET_MODELS[1]}). Continuing..."
fi


log_info "All required downloads attempted."

# --- Launch A1111 ---

# Enforce required arguments, adding them if not present in user args
ENFORCED_ARGS="--xformers --api --no-half-vae"
FINAL_ARGS="$USER_A1111_ARGS"

log_info "Base User Args: $USER_A1111_ARGS"
log_info "Enforcing Args: $ENFORCED_ARGS"

for arg in $ENFORCED_ARGS; do
  # Check if the argument or its prefix exists (e.g., --api vs --api=foo)
  if ! echo " $FINAL_ARGS " | grep -q " $arg "; then
      # Check if a variation with '=' exists
      if ! echo " $FINAL_ARGS " | grep -q " ${arg}= "; then
          log_info "Adding enforced argument: $arg"
          FINAL_ARGS="$FINAL_ARGS $arg"
      else
          log_info "Enforced argument '$arg' seems present with a value."
      fi
  else
      log_info "Enforced argument '$arg' already present."
  fi
done

# Remove potential leading/trailing whitespace
FINAL_ARGS=$(echo "$FINAL_ARGS" | awk '{$1=$1};1')

log_info "Final effective A1111 launch arguments: $FINAL_ARGS"

# Start the WebUI
log_info "Changing directory to $A1111_DIR"
cd $A1111_DIR

log_info "Starting Stable Diffusion WebUI via launch.py..."
# Use exec to replace the shell process with the python process
exec python launch.py $FINAL_ARGS

# Log end is typically not reached if exec is successful
log_info "--- Log End: $(date -u) ---"
