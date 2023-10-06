# Step 0. Activate virtual environment and set up env variables
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

# Step 1. Model compilation

echo "Compiling model...May take a few seconds..."
python -m mlc_llm.build \
	--model $PATH_COMPILE \
	--artifact-path $PATH_COMPILE \
	--quantization $QUANTIZATION \
	--max-seq-len 2048 \
	--num-shards $NUM_SHARDS \
	--target cuda --use-cuda-graph --build-model-only
mv $PATH_COMPILE/model-${QUANTIZATION}/model-${QUANTIZATION}-cuda.so $PATH_TEST/${MODEL_NAME}-${QUANTIZATION}-cuda.so

# Step 2. Benchmarking

echo "benchmarking..."
python -m mlc_chat.cli.benchmark \
	--model ${PATH_TEST}/params \
	--device "cuda:0" \
	--prompt "What is the meaning of life?" \
	--generate-length 256
