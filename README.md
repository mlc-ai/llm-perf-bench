LLM Performance Benchmarking
----------------------------

## Performance

### Int4-quantized, Single GPU

| Model      | GPU         | MLC LLM (tok/sec) | Exllama V2 (tok/sec) | Llama.cpp (tok/sec) |
|------------|-------------|-------------------|----------------------|---------------------|
| Llama2-7B  | RTX 3090 Ti | 186.7             | 161.67               | 134.54              |
| Llama2-13B | RTX 3090 Ti | 107.4             | 92.11                | 81.48               |
| Llama2-7B  | RTX 4090    | 204.8             | 177.46               | 151.1               |
| Llama2-13B | RTX 4090    | 113.5             | 105.94                | 88.0                |

All experiments are based on int4-quantized weights, fp16 activation and compute, decoding for 256 tokens with a prompt "What is the meaning of life?".

### FP16, Multi-GPU

TBD

## Instructions

### Prerequisites

**GPU Docker**. Before proceeding, make sure you have NVIDIA Docker installed for NVIDIA GPUs. Follow the installation guide at [NVIDIA Docker Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker) for detailed instructions.

<table>
<tr>
<th> CUDA </th>
<th> ROCm </th>
</tr>
<tr>
<td>

```bash
docker run --gpus all \
  nvidia/cuda:12.1.1-devel-ubuntu22.04 nvidia-smi
```

</td>
<td>

```bash
docker run --device=/dev/kfd --device=/dev/dri   \
           --security-opt seccomp=unconfined     \
           --group-add video \
       rocm/rocm-terminal rocm-smi
```

</td>
</tr>
</table>

**Repository Setup**. Clone the repository, as all subsequent steps assume you are in the repository root:

```bash
git clone https://github.com/mlc-ai/llm-perf-bench
cd llm-perf-bench
```

Now you are ready to proceed with the next steps in the repository.

---

### MLC LLM

In this section, we use int4 quantized Llama2 as an example.

**Step 1**. Build Docker image and download pre-quantized weights from HuggingFace, then log into the docker image and activate Python environment:

<details>

```bash
git lfs install
git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1
# git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-13b-chat-hf-q4f16_1
# git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-70b-chat-hf-q4f16_1
# git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-70b-chat-hf-q0f16
# git clone https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-7b-Instruct-hf-q4f16_1
# git clone https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-13b-Instruct-hf-q4f16_1
# git clone https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-34b-Instruct-hf-q4f16_1
# git clone https://huggingface.co/mlc-ai/mlc-chat-CodeLlama-34b-Instruct-hf-q0f16
```

<table>
<tr>
<th> CUDA </th>
<th> ROCm </th>
</tr>
<tr>
<td>

```bash
docker build --no-cache -t llm-perf-mlc:v0.1    \
    -f ./docker/Dockerfile.cu121.mlc .
./docker/bash.sh llm-perf-mlc:v0.1

```

</td>
<td>

```bash
docker build --no-cache -t llm-perf-mlc:v0.1    \
    -f ./docker/Dockerfile.rocm57.mlc .
./docker/bash.sh --amd llm-perf-mlc:v0.1
```

</td>
</tr>
</table>

</details>

**Step 2**. Stay logged in, set some basic environment variables for convenient scripting.

<details>

```bash
conda activate python311
MODEL_NAME=Llama-2-7b-chat-hf
QUANTIZATION=q4f16_1
NUM_SHARDS=1
PATH_COMPILE=/tmp/model/
PATH_TEST=/tmp/test/

MODEL_CONFIG=./model_configs/${MODEL_NAME}.json
WEIGHT_PATH=$(pwd)/mlc-chat-${MODEL_NAME}-${QUANTIZATION}/

if [ -e "$WEIGHT_PATH/mlc-chat-config.json" ]; then
	sed -i "/\"num_shards\"/c\ \"num_shards\": ${NUM_SHARDS}," $WEIGHT_PATH/mlc-chat-config.json
else
	echo "Path '$WEIGHT_PATH/mlc-chat-config.json' does not exist."
	exit
fi

rm -rf $PATH_TEST && mkdir $PATH_TEST && rm -rf $PATH_COMPILE && mkdir $PATH_COMPILE && ln -s ${WEIGHT_PATH} ${PATH_TEST}/params && cp $MODEL_CONFIG $PATH_COMPILE/config.json
```

</details>

**Step 3**. Stay logged in, and compile MLC model lib. It may take a few seconds:

<details>

<table>
<tr>
<th> CUDA </th>
<th> ROCm </th>
</tr>
<tr>
<td>

```bash
python -m mlc_llm.build \
	--model $PATH_COMPILE \
	--artifact-path $PATH_COMPILE \
	--quantization $QUANTIZATION \
	--max-seq-len 2048 \
	--num-shards $NUM_SHARDS \
	--target cuda --use-cuda-graph --build-model-only
mv $PATH_COMPILE/model-${QUANTIZATION}/model-${QUANTIZATION}-cuda.so \
                    $PATH_TEST/${MODEL_NAME}-${QUANTIZATION}-cuda.so
```

</td>
<td>

```bash
python -m mlc_llm.build \
	--model $PATH_COMPILE \
	--artifact-path $PATH_COMPILE \
	--quantization $QUANTIZATION \
	--max-seq-len 2048 \
	--num-shards $NUM_SHARDS \
	--target rocm --build-model-only
mv $PATH_COMPILE/model-${QUANTIZATION}/model-${QUANTIZATION}-rocm.so \
                    $PATH_TEST/${MODEL_NAME}-${QUANTIZATION}-rocm.so
```

</td>
</tr>
</table>

</details>

**Step 4**. Stay logged in, and run benchmarking:

<details>

<table>
<tr>
<th> CUDA </th>
<th> ROCm </th>
</tr>
<tr>
<td>

```bash
python -m mlc_chat.cli.benchmark \
	--model ${PATH_TEST}/params \
	--device "cuda:0" \
	--prompt "What is the meaning of life?" \
	--generate-length 256
```

</td>
<td>

```bash
python -m mlc_chat.cli.benchmark \
	--model ${PATH_TEST}/params \
	--device "rocm:0" \
	--prompt "What is the meaning of life?" \
	--generate-length 256
```

</td>
</tr>
</table>

</details>

---

### Exllama V2

In this section, we use Llama2 GPTQ model as an example.

**Step 1**. Build Docker image and download pre-quantized weights from HuggingFace, then log into the docker image and activate Python environment:

<details>

```bash
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-7B-GPTQ
# git clone https://huggingface.co/TheBloke/Llama-2-70B-chat-GPTQ
# git clone https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GPTQ

docker build --no-cache -t llm-perf-exllama-v2:v0.1    \
    -f ./docker/Dockerfile.cu121.exllama_v2 .
./docker/bash.sh llm-perf-exllama-v2:v0.1
conda activate python311
```

</details>

**NOTE**. Docker image building for ExllamaV2 is particularly memory consuming on certain GPU instances.
Kill the process in time if it lags or screen freezes.

**Step 2**. Stay logged in, run benchmarking

<details>

For single GPU:
```bash
MODEL_PATH=/workspace/Llama-2-7B-GPTQ/
OUTPUT_LEN=256
cd /exllamav2
python test_inference.py -m $MODEL_PATH -p "What is the meaning of life?" -t $OUTPUT_LEN
```

For Multiple GPU:
```bash
MODEL_PATH=$(pwd)/Llama-2-7B-GPTQ/
OUTPUT_LEN=256
GPU_SPLIT="17,17" # depend on how you want to split memory
cd /exllamav2
python test_inference.py -m $MODEL_PATH -p "What is the meaning of life?" -gs $GPU_SPLIT -t $OUTPUT_LEN
```

</details>

---

### Llama.cpp

**Step 1**. Build Docker image:

<details>

```bash
docker build --no-cache -t llm-perf-llama-cpp:v0.1 -f ./docker/Dockerfile.cu121.llama_cpp .
```

</details>

**Step 2**. Download the quantized GGML models from HuggingFace:

<details>

```bash
mkdir -p ./llama_cpp_models
wget -O ./llama_cpp_models/llama-2-7b.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
wget -O ./llama_cpp_models/llama-2-70b.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-70B-GGUF/resolve/main/llama-2-70b.Q4_K_M.gguf
wget -O ./llama_cpp_models/codellama-34b.Q4_K_M.gguf https://huggingface.co/TheBloke/CodeLlama-34B-GGUF/resolve/main/codellama-34b.Q4_K_M.gguf
# wget -O ./llama_cpp_models/llama-2-13b.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_M.gguf
# wget -O ./llama_cpp_models/llama-2-70b.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-70B-GGUF/resolve/main/llama-2-70b.Q4_K_M.gguf
```

</details>

**Step 3**. Log into docker, and the CLI tool to see the performance numbers. Note that modify `CUDA_VISIBLE_DEVICES` settings for different numbers of GPUs experiments.

<details>

```bash
./docker/bash.sh llm-perf-llama-cpp:v0.1
cd /llama.cpp
# run quantized Llama-2-7B models on a single GPU.
CUDA_VISIBLE_DEVICES=0 ./build/bin/main -m /workspace/llama_cpp_models/llama-2-7b.Q4_K_M.gguf -p "What is the meaning of life?" -n 256 -ngl 999 --ignore-eos
# test quantized Llama-2-70B models on 2 GPUS.
CUDA_VISIBLE_DEVICES=0,1 ./build/bin/main -m /workspace/llama_cpp_models/llama-2-70b.Q4_K_M.gguf -p "What is the meaning of life?" -n 256 -ngl 999 --ignore-eos
```

</details>

**Note**. For float16 models, stay logged in and convert the hf models (download [here](https://huggingface.co/meta-llama/Llama-2-70b-hf)) to GGUF FP16 format first.

<details>

```bash
cd /llama.cpp
conda activate python311
# convert the weight using llama.cpp script
python3 convert.py /path/to/Llama-2-70b-hf/ \
    --outfile /workspace/llama_cpp_models/llama-2-70b.fp16.gguf
# run fp16 models on 4 GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/bin/main -m /workspace/llama_cpp_models/llama-2-70b.fp16.gguf -p "What is the meaning of life?" -n 256 -ngl 999 --ignore-eos
```

</details>

---

### HuggingFace Transformer

**Step 1**. Build Docker image:

<details>

```bash
docker build -t llm-perf-hf:v0.1 -f ./docker/Dockerfile.cu121.hf .
```

</details>

**Step 2**. Download Llama-2 weight from [huggingface](https://huggingface.co/meta-llama/Llama-2-70b-hf).

<details>

```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
# git clone https://huggingface.co/meta-llama/Llama-2-13b-hf
# git clone https://huggingface.co/meta-llama/Llama-2-70b-hf
```

</details>

**Step 3**. Log into docker and run the python script to see the performance numbers. Note that modify `CUDA_VISIBLE_DEVICES` settings for different numbers of GPUs experiments:

<details>

```bash
./docker/bash.sh llm-perf-hf:v0.1
conda activate python311
# run fp16 Llama-2-7b models on a single GPU.
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_hf.py --model-path ./Llama-2-7b-hf --format q0f16 --prompt "What is the meaning of life?" --max-new-tokens 256
# run int 4 quantized Llama-2-70b model on two GPUs.
CUDA_VISIBLE_DEVICES=0,1 python scripts/benchmark_hf.py --model-path ./Llama-2-70b-hf --format q4f16 --prompt "What is the meaning of life?" --max-new-tokens 256
```

</details>

### vLLM

In this section, we use Llama2 GPTQ model as an example.

**Step 1**. Build Docker image and download pre-quantized weights from HuggingFace, then log into the docker image and activate Python environment:
<details>

```bash
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-70B-fp16
docker build --no-cache -t llm-perf-vllm:v0.1    \
    -f ./docker/Dockerfile.cu118.vllm .
./docker/bash.sh llm-perf-vllm:v0.1
conda activate python311
```
</details>
**Step 2**. Modify script and run benchmarking

TODO: Fix script for single batch & allow prompt input instead of dummy

<details>

For single GPU:
```bash
MODEL_PATH=$(pwd)/Llama-2-7B-fp16/
OUTPUT_LEN=256
cd /vllm && python benchmarks/benchmark_latency.py \
--model $MODEL_PATH \
--output-len $OUTPUT_LEN \
--batch-size 1 \
--input-len 7 # for prompt "What is the meaning of life?"
```

For multiple GPUs:
```bash
MODEL_PATH=$(pwd)/Llama-2-7B-fp16/
OUTPUT_LEN=256
GPU_NUM=2
cd /vllm && python benchmarks/benchmark_latency.py \
--model $MODEL_PATH \
--output-len $OUTPUT_LEN \
--tensor-parallel-size $GPU_NUM \
--batch-size 1 \
--input-len 7 # for prompt "What is the meaning of life?"
```

</details>

## Setup Details

We are using the following commits:
- MLC LLM [commit](https://github.com/mlc-ai/mlc-llm/commits/8e94910ec7967cbe749dbf04713f96a52cccbc19), TVM [commit](https://github.com/mlc-ai/relax/commits/e5ca38dd735ba4d30782a4a58bf6195861642eb0) on 10/04/2023;
- ExllamaV2 [commit](https://github.com/turboderp/exllamav2/commits/9d6fdb952f6705f79415364e9d85989dcda01478) on 10/05/2023;
- Llama.cpp [commit](https://github.com/ggerganov/llama.cpp/commits/9476b012260a2fb6c67976582d64484ce7406ed9) on 10/02/2023;
- vLLM [commit](https://github.com/vllm-project/vllm/commit/acbed3ef40f015fcf64460e629813922fab90380) on 10/06/2023;
- HuggingFace transformers 4.33.3 on 10/06/2023.
