#!/usr/bin/env bash
#
# Start a bash, mount /workspace to be current directory.
#
# Usage: docker/bash.sh <CONTAINER_NAME>
#     Starts an interactive session
#
# Usage2: docker/bash.sh <CONTAINER_NAME> [COMMAND]
#     Execute command in the docker image, non-interactive
#
if [ "$#" -lt 1 ]; then
	echo "Usage: docker/bash.sh <CONTAINER_NAME> [--no-gpu] [COMMAND]"
	exit -1
fi

if [ "$1" == "--no-gpu" ]; then
	ENABLE_GPU=""
	shift
elif [ "$1" == "--amd" ]; then
	ENABLE_GPU="amd"
	shift
else
	ENABLE_GPU="nv"
fi

DOCKER_IMAGE_NAME=("$1")

if [ "$#" -eq 1 ]; then
	COMMAND="bash"
	if [[ $(uname) == "Darwin" ]]; then
		# Docker's host networking driver isn't supported on macOS.
		# Use default bridge network and expose port for jupyter notebook.
		DOCKER_EXTRA_PARAMS=("-it -p 8888:8888")
	else
		DOCKER_EXTRA_PARAMS=("-it --net=host")
	fi
else
	shift 1
	COMMAND=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(pwd)"

if [ "$ENABLE_GPU" == "nv" ]; then
	if ! type "nvidia-docker" 1>/dev/null 2>/dev/null; then
		DOCKER_BINARY="docker"
		# Fix for NCCL: https://github.com/NVIDIA/nccl-tests/issues/143
		CUDA_ENV=" --gpus all --shm-size 16G "${CUDA_ENV}
	else
		DOCKER_BINARY="nvidia-docker"
	fi
elif [ "$ENABLE_GPU" == "amd" ]; then
	CUDA_ENV=" --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video"
	DOCKER_BINARY="docker"
else
	DOCKER_BINARY="docker"
fi

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "DOCKER CONTAINER NAME: ${DOCKER_IMAGE_NAME}"
echo ""

echo "Running '${COMMAND[@]}' inside ${DOCKER_IMAGE_NAME}..."

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).

${DOCKER_BINARY} run --rm --pid=host --shm-size 10.24g -v ${WORKSPACE}:/workspace \
	-v ${SCRIPT_DIR}:/docker \
	-w /workspace \
	${CUDA_ENV} \
	${DOCKER_EXTRA_PARAMS[@]} \
	${DOCKER_IMAGE_NAME} \
	${COMMAND[@]}
