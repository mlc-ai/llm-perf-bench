LLM Performance Benchmarking
----------------------------

## Performance

| Model      | GPU         | MLC LLM (tok/sec) | Exllama (tok/sec) | Llama.cpp (tok/sec) |
|------------|-------------|-------------------|-------------------|---------------------|
| Llama2-7B  | RTX 3090 Ti | 166.7             | 112.72            | 123.75              |
| Llama2-13B | RTX 3090 Ti | 99.2              | 69.31             | 76.18               |
| Llama2-7B  | RTX 4090    | 191.0             | 152.56            | 50.13               |
| Llama2-13B | RTX 4090    | 108.8             | 93.88             | 36.81               |

All experiments are based on int4-quantized weights, fp16 activation and compute.

Commit:
- MLC LLM [commit](https://github.com/mlc-ai/mlc-llm/commit/502f6808b8073b87e561817a5a80b50810ab47be), TVM [commit](https://github.com/apache/tvm/commit/543838303b4289bb5669688efb9f88b15ddc2ebe);
- Exllama [commit](https://github.com/turboderp/exllama/commit/c16cf49c3f19e887da31d671a713619c8626484e).
- Llama.cpp: [commit](https://github.com/ggerganov/llama.cpp/commit/9476b012260a2fb6c67976582d64484ce7406ed9)


## Instructions

First of all, NVIDIA Docker is required: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker.

### MLC LLM

**Step 1**. Build Docker image

```bash
docker build -t llm-perf-mlc:v0.1 -f Dockerfile.cu121.mlc .
```

**Step 2**. Quantize and run Llama2. Log in to the docker container we created using the command below:

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

To obtain the quantized GGUF model, it is recommended to download it via HuggingFace using
the command below(replace `/PATH/TO/MODELS` with your custom path):

```bash
wget -O /PATH/TO/MODELS/llama-2-7b.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf 
wget -O /PATH/TO/MODELS/llama-2-13b.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_M.gguf
wget -O /PATH/TO/MODELS/llama-2-70b.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-70B-GGUF/resolve/main/llama-2-70b.Q4_K_M.gguf
```

To test on float16 format, download Llama-2-70b-hf weight from [huggingface](https://huggingface.co/meta-llama/Llama-2-70b-hf), and place the model folder under your `/PATH/TO/MODELS`.

```bash
PORT=41514
MODEL_PATH=/PATH/TO/MODELS/  # Replace this with the path to the directory containing the HuggingFace models

docker run                  \
  -d -P                     \
  --gpus all                \
  -h llm-perf               \
  --name llm-perf-llama-cpp \
  -p $PORT:22               \
  -v $MODEL_PATH:/models  \
  llm-perf-llama-cpp:v0.1

# Password is: llm_perf
ssh root@0.0.0.0 -p $PORT
```

**Step 3.** Run the CLI tool to see the performance numbers:

Log in to the docker container we created using the command below:

```bash
cd $LLAMA_CPP_HOME
# run quantized models
CUDA_VISIBLE_DEVICES=0 ./build/bin/main -m /models/llama-2-7b.Q4_K_M.gguf -p "Please generate a very long story about wizard and technology, at least two thousand words" -n 128 -ngl 999 --ignore-eos
# test quantized 70B models on 2 gpus
CUDA_VISIBLE_DEVICES=0,1 ./build/bin/main -m /models/llama-2-70b.Q4_K_M.gguf -p "Please generate a very long story about wizard and technology, at least two thousand words" -n 128 -ngl 999 --ignore-eos
```

To evaluate the performance of the float16 model, please convert the hf models to GGUF FP16 format first.
```bash
cd $LLAMA_CPP_HOME
python3 convert.py /models/Llama-2-70b-hf/ --outfile /models/llama-2-70b.fp16.gguf
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/bin/main -m /models/llama-2-70b.fp16.gguf -p "Please generate a very long story about wizard and technology, at least two thousand words" -n 128 -ngl 999 --ignore-eos
```


### HuggingFace

**Step 1**. Build Docker image

```bash
docker build -t llm-perf-hf:v0.1 -f Dockerfile.cu121.hf .
```

**Step 2**. Download the HuggingFace models and run Llama2 via script.

Download Llama-2-70b-hf weight from [huggingface](https://huggingface.co/meta-llama/Llama-2-70b-hf).

```bash
PORT=41598
MODEL_WEIGHTS=/PATH/TO/Llama-2-70b-hf/  # Replace this with the path to the directory containing the HuggingFace models

docker run                   \
  -d -P                      \
  --gpus all                 \
  -h llm-perf                \
  --name llm-perf-hf         \
  -p $PORT:22                \
  -v $MODEL_WEIGHTS:/models  \
  llm-perf-hf:v0.1

# Password is: llm_perf
ssh root@0.0.0.0 -p $PORT
```

**Step 3.** Run the python script to see the performance numbers:

Log in to the docker container we created using the command below:

```bash
micromamba activate python311
# test float16 model
python benchmark_hf.py --model-path /models/Llama-2-70b-hf --format q0f16
# test 4-bit quantized model
python benchmark_hf.py --model-path /models/Llama-2-70b-hf --format q4f16
```

## TODOs

Only decoding performance is currently benchmarked given prefilling usually takes much shorter time with flash attention.

Currently, MLC LLM number includes a [long system prompt](https://github.com/mlc-ai/mlc-llm/blob/c40be6a210e4d8844b8a65951bcfaa44b528b8f9/cpp/conv_templates.cc#L35),
while Exllama numbers are from a fixed-length system prompt of 4 tokens,
which is not exactly apple-to-apple comparison. Should get it fixed.
