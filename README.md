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

**Step 1**. Build Docker image

```bash
docker build -t llm-perf-mlc:v0.1 -f Dockerfile.cu121.mlc .
```

**Step 2**. Quantize and run Llama2. Log in to the docker container we created using the comamnd below:

```bash
PORT=45678
MODELS=/PATH/TO/MODEL/ # Replace the path to HuggingFace models

docker run            \
  -d -P               \
  --gpus all          \
  -h llm-perf         \
  --name llm-perf     \
  -p $PORT:22         \
  -v $MODELS:/models  \
  llm-perf-mlc:v0.1

# Password is: llm_perf
ssh root@0.0.0.0 -p $PORT

# Inside the container, run the following commands:
micromamba activate python311

cd $MLC_HOME
python build.py                       \
  --model /models/Llama-2-7b-chat-hf  \  # Replace it with path to HuggingFace models
  --target cuda                       \
  --quantization q4f16_1              \
  --artifact-path "./dist"            \
  --use-cache 0
```

The quantized and compiled model will be exported to `./dist/Llama-2-7b-chat-hf-q4f16_1`.

**Step 3.** Run the Python bechmarking scripts according to "examples/python
/benchmark.py".


### Exllama

TBD

### Llama.cpp

**Step 1**. Build Docker image

```bash
docker build -t llm-perf-llama-cpp:v0.1 -f Dockerfile.cu121.llama_cpp .
```

**Step 2**. Download the quantized GGML models and run Llama2 via llama.cpp.

To obtain the quantized GGML model, it is recommended to download it via HuggingFace using
the comamnd below:

```bash
wget https://huggingface.co/TheBloke/Llama-2-7B-GGML/resolve/main/llama-2-7b.ggmlv3.q4_K_M.bin
wget https://huggingface.co/TheBloke/Llama-2-13B-GGML/resolve/main/llama-2-13b.ggmlv3.q4_K_M.bin
```

```bash
PORT=41514
GGML_BINS=/PATH/TO/GGML_BINS/  # Replace it with path to HuggingFace models

docker run                  \
  -d -P                     \
  --gpus all                \
  -h llm-perf               \
  --name llm-perf-llama-cpp \
  -p $PORT:22               \
  -v $GGML_BINS:/ggml_bins  \
  llm-perf-llama-cpp:v0.1

# Password is: llm_perf
ssh root@0.0.0.0 -p $PORT
```

**Step 3.** Run the CLI tool to see the performance numbers:

Log in to the docker container we created using the comamnd below:

```bash
cd $LLAMA_CPP_HOME
./build/bin/main -m /ggml_bins/llama-2-7b.ggmlv3.q4_K_M.bin -p "Please generate a very long story about wizard and technology, at least two thousand words" -n 128 -ngl 999 --ignore-eos
```

## TODOs

Only decoding performance is currently benchmarked given prefilling usually takes much shorter time with flash attention.

Currently, MLC LLM number includes a [long system prompt](https://github.com/mlc-ai/mlc-llm/blob/c40be6a210e4d8844b8a65951bcfaa44b528b8f9/cpp/conv_templates.cc#L35),
while Exllama numbers are from a fixed-length system prompt of 4 tokens,
which is not exactly apple-to-apple comparison. Should get it fixed.
