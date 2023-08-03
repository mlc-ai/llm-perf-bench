# This Dockerfile is based on CUDA 12.1.
# Please search and replace "12.1"/"121" if benchmarking on other CUDA versions.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-ec"]

# Step 1. Set up Ubuntu
RUN grep -v '[ -z "\$PS1" ] && return' ~/.bashrc >/tmp/bashrc               && \
    mv /tmp/bashrc ~/.bashrc                                                && \
    echo "export MLC_HOME=/mlc_llm/"                                        && \
    echo "export PATH=/usr/local/cuda/bin/:\$PATH" >>~/.bashrc              && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:\$PATH" >>~/.bashrc && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so                               \
          /usr/local/cuda/lib64/stubs/libcuda.so.1                          && \
    apt update                                                              && \
    apt install --yes wget curl git vim build-essential openssh-server

# Step 2. Set up python, including TVM Unity
RUN bash <(curl -L micro.mamba.pm/install.sh) && source ~/.bashrc       && \
    micromamba create --yes -n python311 -c conda-forge                    \
    python=3.11 "cmake>=3.24"                                              \
    pytorch-cpu rust sentencepiece protobuf
RUN source ~/.bashrc && micromamba activate python311                   && \
    pip install --pre mlc-ai-nightly-cu121 -f https://mlc.ai/wheels

# Step 3. Compile MLC command line
RUN source ~/.bashrc && micromamba activate python311                   && \
    git clone --recursive https://github.com/mlc-ai/mlc-llm/ $MLC_HOME  && \
    cd $MLC_HOME && mkdir build && cd build && touch config.cmake       && \
    echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >>config.cmake          && \
    echo "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" >>config.cmake         && \
    echo "set(USE_CUDA ON)" >>config.cmake                              && \
    echo "set(USE_VULKAN OFF)" >>config.cmake                           && \
    echo "set(USE_METAL OFF)" >>config.cmake                            && \
    echo "set(USE_OPENCL OFF)" >>config.cmake                           && \
    cmake .. && make -j$(nproc)

# Step 4. Set up SSH and expose SSH port
COPY install/ssh.sh /install/ssh.sh
RUN bash /install/ssh.sh
WORKDIR /root
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
