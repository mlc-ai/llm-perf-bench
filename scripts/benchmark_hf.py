import argparse
import copy
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import LogitsProcessorList, StoppingCriteriaList
from transformers.models.llama import LlamaForCausalLM

COMPILE = False


class Recorder:
    def __init__(self) -> None:
        self.time = {}
        self.cnt = {}

    def record(self, name, num) -> None:
        if name not in self.time:
            self.time[name] = 0.0
            self.cnt[name] = 0
        self.time[name] += num
        self.cnt[name] += 1

    def get(self, name):
        return self.time[name], self.cnt[name]


class Timer:
    def __init__(self, name: str, recorder: Recorder):
        self.name = name
        self.recorder = recorder
        self._start_time = None
        self._end_time = None

    def __enter__(self):
        torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def __exit__(self, *exc_info):
        torch.cuda.synchronize()
        self._end_time = time.perf_counter()
        elapsed_time = self._end_time - self._start_time
        self.recorder.record(self.name, elapsed_time)


def create_fp16(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, use_safetensors=False
    )
    if COMPILE:
        model = torch.compile(model, dynamic=True)
    return model


def create_q4fp16(model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        use_safetensors=False,
    )
    if COMPILE:
        model = torch.compile(model, dynamic=True)
    return model


@torch.no_grad()
def sample(model: LlamaForCausalLM, perf_recorder: Recorder, input_ids, **kwargs):
    this_peer_finished = False
    generation_config = copy.deepcopy(model.generation_config)
    model_kwargs = generation_config.update(**kwargs)

    input_ids_length = input_ids.shape[-1]
    generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=model.config.is_encoder_decoder,
        **model_kwargs,
    )
    logits_warper = model._get_logits_warper(generation_config)
    logits_processor = LogitsProcessorList()
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=StoppingCriteriaList()
    )
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if model_inputs["input_ids"].shape[-1] > 1:
            with Timer("prefill", perf_recorder):
                outputs = model(**model_inputs, return_dict=True)
        else:
            with Timer("decode", perf_recorder):
                outputs = model(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )

        if stopping_criteria(input_ids, None):
            this_peer_finished = True

        if this_peer_finished:  # and not synced_gpus:
            break

    return input_ids


def log_perf(name, elapsed_time, cnt):
    token_per_sec = cnt / elapsed_time
    latency = elapsed_time / cnt

    return f"{name} {elapsed_time:.4g} s / {cnt} {'tokens' if name == 'prefill' else 'runs'} ({latency:.4g} s/tok, {token_per_sec:.4g} tok/s)"


def main(ARGS):
    model_id = ARGS.model_path
    max_new_tokens = ARGS.max_new_tokens
    quantize = ARGS.format
    prompt = ARGS.prompt

    print(
        f"Using model from {model_id}, max_new_tokens = {max_new_tokens}, format = {quantize}, prompt = {prompt}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quantize == "q0f16":
        model = create_fp16(model_id)
    elif quantize == "q4f16":
        model = create_q4fp16(model_id)
    else:
        assert False

    input_text = prompt
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    recorder = Recorder()
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        outputs = sample(
            model, recorder, inputs["input_ids"], max_new_tokens=max_new_tokens
        )

    prefill_elapsed_time, _ = recorder.get("prefill")
    decode_elapsed_time, decode_num = recorder.get("decode")
    assert decode_num == max_new_tokens - 1

    print("=" * 10)
    print(log_perf("prefill", prefill_elapsed_time, inputs["input_ids"].shape[-1]))
    print(log_perf("decode", decode_elapsed_time, decode_num))
    print("=" * 10)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--format", choices=["q0f16", "q4f16"])
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--prompt", type=str)

    ARGS = parser.parse_args()
    main(ARGS)
