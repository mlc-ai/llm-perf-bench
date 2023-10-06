LLM Performance Benchmarking
----------------------------

## Performance

| Model      | GPU         | MLC LLM (tok/sec) | Exllama (tok/sec) | Llama.cpp (tok/sec) |
|------------|-------------|-------------------|-------------------|---------------------|
| Llama2-7B  | RTX 3090 Ti | 166.7             | 112.72            | 113.34              |
| Llama2-13B | RTX 3090 Ti | 99.2              | 69.31             | 71.34               |
| Llama2-7B  | RTX 4090    | 191.0             | 152.56            | 50.13               |
| Llama2-13B | RTX 4090    | 108.8             | 93.88             | 36.81               |

All experiments are based on int4-quantized weights, fp16 activation and compute.

Commit:
- MLC LLM [commit](https://github.com/mlc-ai/mlc-llm/commit/502f6808b8073b87e561817a5a80b50810ab47be), TVM [commit](https://github.com/apache/tvm/commit/543838303b4289bb5669688efb9f88b15ddc2ebe);
- Exllama [commit](https://github.com/turboderp/exllama/commit/c16cf49c3f19e887da31d671a713619c8626484e).
- Llama.cpp: [commit](https://github.com/ggerganov/llama.cpp/commit/f3c3b4b1672d860800639c87d3b5d17564692469)

## Instructions

First of all, NVIDIA Docker is required: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker.

### MLC LLM

In this section, we use int4 quantized Llama2 as an example.

**Step 1**. Build Docker image and download pre-quantized weights from HuggingFace:

<details>

```bash
docker build -t llm-perf-mlc:v0.1 -f ./docker/Dockerfile.cu121.mlc .
git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1
git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-13b-chat-hf-q4f16_1
git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-70b-chat-hf-q4f16_1
```

</details>

**Step 2.** Log into docker, activate Python environment, and set some basic environment variables for convenient scripting.

<details>

```bash
./docker/bash.sh llm-perf-mlc:v0.1

conda activate python311

MODEL_CONFIG=./model_configs/llama2_7b.json
QUANTIZATION=q4f16_1
MODEL_NAME=Llama-2-7b-chat-hf
NUM_SHARDS=1
WEIGHT_PATH=$(pwd)/mlc-chat-${MODEL_NAME}-${QUANTIZATION}/
PATH_COMPILE=/tmp/model/
PATH_TEST=/tmp/test/

if [ -e "$WEIGHT_PATH/mlc-chat-config.json" ]; then
	sed -i "/\"num_shards\"/c\ \"num_shards\": ${NUM_SHARDS}," $WEIGHT_PATH/mlc-chat-config.json
else
	echo "Path '$WEIGHT_PATH/mlc-chat-config.json' does not exist."
	exit
fi

rm -rf $PATH_TEST && mkdir $PATH_TEST && rm -rf $PATH_COMPILE && mkdir $PATH_COMPILE
ln -s ${WEIGHT_PATH} ${PATH_TEST}/params
cp $MODEL_CONFIG $PATH_COMPILE/config.json
```

</details>

**Step 3.** Stay logged in, and compile MLC model lib. It may take a few seconds:

<details>

```bash
python -m mlc_llm.build \
	--model $PATH_COMPILE \
	--artifact-path $PATH_COMPILE \
	--quantization $QUANTIZATION \
	--max-seq-len 2048 \
	--num-shards $NUM_SHARDS \
	--target cuda --use-cuda-graph --build-model-only
mv $PATH_COMPILE/model-${QUANTIZATION}/model-${QUANTIZATION}-cuda.so $PATH_TEST/${MODEL_NAME}-${QUANTIZATION}-cuda.so
```

</details>

**Step 4.** Run benchmarking:

<details>

```bash
echo "benchmarking..."
python -m mlc_chat.cli.benchmark \
	--model ${PATH_TEST}/params \
	--device "cuda:0" \
	--prompt "What is the meaning of life?" \
	--generate-length 256
```

</details>

### Exllama

TBD

### Llama.cpp

TBD
