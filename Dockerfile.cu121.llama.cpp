# NOTE: This Dockerfile is based on CUDA 12.1.
# To benchmark on other CUDA versions, search and replace "12.1" and "121".
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-ec"]

# Step 1. Set up Ubuntu
# NOTE: libcuda.so.1 doesn't exist in NVIDIA's base image, link the stub file to work around
RUN grep -v '[ -z "\$PS1" ] && return' ~/.bashrc >/tmp/bashrc                                 && \
    mv /tmp/bashrc ~/.bashrc                                                                  && \
    echo "export LLAMA_CPP_HOME=/llama.cpp/" >>~/.bashrc                                              && \
    echo "export PATH=/usr/local/cuda/bin/:\$PATH" >>~/.bashrc                                && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1       && \
    apt update                                                                                && \
    apt install --yes wget curl git vim build-essential openssh-server cmake

# Step 2. Set up python environment with micromamba
RUN bash <(curl -L micro.mamba.pm/install.sh) && source ~/.bashrc       && \
    micromamba create --yes -n python311 -c conda-forge                    \
    python=3.11 "cmake>=3.24"                                              \
    pytorch-cpu rust sentencepiece protobuf                             && \
    micromamba activate python311

# Step 3. Git clone and compile llama.cpp with cuBLAS
RUN git clone https://github.com/ggerganov/llama.cpp.git $LLAMA_CPP_HOME && \
    cd llama.cpp                                                         && \
    git checkout f3c3b4b1672d860800639c87d3b5d17564692469                && \
    make LLAMA_CUBLAS=1                                                  && \
    mkdir build                                                          && \
    cd build                                                             && \
    cmake .. -DLLAMA_CUBLAS=1                                            && \
    cmake --build . --config Release

# Step 4. Set up SSH and clean up
COPY install/ssh.sh /install/ssh.sh
RUN bash /install/ssh.sh && rm /usr/lib/x86_64-linux-gnu/libcuda.so.1
WORKDIR /root
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
