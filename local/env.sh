# Standalone NeMo AutoModel dev environment for NVIDIA-NeMo/Automodel#4122 on cloudml.
# Usage: source /planning-users/yangyeyuan/automodel-4122/env.sh  (cmdev run/exec does this)
#
# Reuses the toolchain built for the NeMo RL workspaces (uv, managed Python, CUDA 13.0,
# uv cache, HF cache) in /planning-users/yangyeyuan/nemo-rl-3976/env. Only the checkout,
# its venv (Automodel/.venv) and logs are specific to this workspace.

SHARED_ENV=/planning-users/yangyeyuan/nemo-rl-3976/env
export AM_WS=/planning-users/yangyeyuan/automodel-4122
export AM_ROOT=$AM_WS/Automodel
mkdir -p "$AM_WS/logs"

# uv + managed Python (Automodel pins 3.12 in .python-version).
export PATH="$SHARED_ENV/tools/uv/bin:$SHARED_ENV/cuda-home/bin:$PATH"
export UV_CACHE_DIR=$SHARED_ENV/uv-cache
export UV_PYTHON_INSTALL_DIR=$SHARED_ENV/python
export UV_PYTHON_INSTALL_MIRROR=<redacted-url>
export UV_HTTP_TIMEOUT=600
export UV_LINK_MODE=hardlink

# CUDA 13.0 toolkit (matches torch cu130 and the 580 driver); /usr/local/cuda is 12.6.
export CUDA_HOME=$SHARED_ENV/cuda-home
export CUDA_PATH=$CUDA_HOME
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=16

# Keep temp files and .pyc off the small root overlay and off JuiceFS.
export TMPDIR=/dev/shm/am-tmp
export PYTHONPYCACHEPREFIX=/dev/shm/am-pycache
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX"

# Hugging Face through the Artifactory HF remote (cache shared with the NeMo RL workspaces).
export HF_HOME=/planning-users/yangyeyuan/nemo-rl-3976/hf
export HF_ENDPOINT=<redacted-url>
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_DISABLE_XET=1

# Artifactory redirects downloads to this storage host.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.mi-fds.net" no_proxy="${no_proxy:+$no_proxy,}.mi-fds.net"

unset SHARED_ENV
