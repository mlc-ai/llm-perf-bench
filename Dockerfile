FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV MLC_HOME /mlc-llm

# Step 1. Set up Ubuntu
RUN apt update && apt install --yes software-properties-common wget git curl vim cmake build-essential python3-pip openssh-server
RUN echo "export PATH=/usr/local/cuda/bin/:\$PATH" >>~/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:\$PATH" >>~/.bashrc

# Step 2. Set up SSH
COPY install/ssh.sh /install/ssh.sh
RUN bash /install/ssh.sh

# Step 3. Set up Python via conda
COPY install/python.sh /install/python.sh
RUN bash /install/python.sh

# Step 4. Set up TVM Unity
COPY install/tvm.sh /install/tvm.sh
RUN git clone --recursive https://github.com/junrushao/mlc-llm/ --branch benchmark $MLC_HOME
RUN bash /install/tvm.sh

# Step 5. Compile MLC command line
COPY install/mlc.sh /install/mlc.sh
RUN bash /install/mlc.sh

# Finally, expose SSH port
WORKDIR /root
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
