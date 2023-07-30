MLC LLM Performance Benchmarking
--------------------------------

## Performance Numbers

| Model      | GPU         | MLC LLM (tok/sec) | Exllama (tok/sec) |
|------------|-------------|-------------------|-------------------|
| Llama2-7B  | RTX 3090 Ti | 154.1             | 116.38            |
| Llama2-13B | RTX 3090 Ti | 93.1              | 70.45             |


Commit:
- MLC LLM: [c40be6a210e4d8844b8a65951bcfaa44b528b8f9](https://github.com/mlc-ai/mlc-llm/tree/c40be6a210e4d8844b8a65951bcfaa44b528b8f9)
- Exllama: [91b9b1295dd9083499fff3d0088c2c1b3c863dc7](https://github.com/turboderp/exllama/tree/91b9b1295dd9083499fff3d0088c2c1b3c863dc7)


## Step-by-step Guide

First of all, NVIDIA Docker is required: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker.

### Step 1. Build Docker image

```bash
docker build -t mlc-perf:v0.1 .
```

### Step 2. Compile and run Llama2

First, log in to the docker container we created using the comamnd below:

```bash
PORT=45678
MODELS=$HOME/models/

docker run            \
  -d -P               \
  --gpus all          \
  -h mlc-perf         \
  --name mlc-perf     \
  -p $PORT:22         \
  -v $MODELS:/models  \
  mlc-perf:v0.1
ssh root@0.0.0.0 -p $PORT # password: mlc_llm_perf
```

> Note: There might be security concerns to allow direct root login. Here we mainly want to simplify the process as a quick demo.

Then, compile Llama2 model using MLC inside the docker container:

```bash
micromamba activate python311

cd $MLC_HOME
python build.py \
  --model /models/Llama-2/hf/Llama-2-7b-chat-hf \
  --target cuda \
  --quantization q4f16_1 \
  --artifact-path "./dist" \
  --use-cache 0
```

The quantized and compiled model will be exported to `./dist/Llama-2-7b-chat-hf-q4f16_1`.

Finally, run the model and see the performance numbers:

```bash
$MLC_HOME/build/mlc_chat_cli \
  --model Llama-2-7b-chat-hf \
  --quantization q4f16_1
```

## TODOs

Only decoding performance is currently benchmarked given prefilling usually takes much shorter time with flash attention.

Currently, MLC LLM number includes a [long system prompt](https://github.com/mlc-ai/mlc-llm/blob/c40be6a210e4d8844b8a65951bcfaa44b528b8f9/cpp/conv_templates.cc#L35),
while Exllama numbers are from a fixed-length system prompt of 4 tokens,
which is not exactly apple-to-apple comparison. Should get it fixed.
