import numpy as np
import torch
import argparse
import time
import json
import sys
import gc
from pathlib import Path
from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
)
from cache_manager import SinkCache, DynamicCache, ShortCache, TcpCache
from tqdm import tqdm
from datasets import load_dataset
from configuration_llama import LlamaConfig
from transformers import AutoModelForCausalLM, AutoConfig

from viz_utils import draw_line_char, write_csv_line

from speculative_inference import SPD
from baseline import Ssd, Lade, Base, CohereForCausalLM

CLASS_MAP = {
    "lade": Lade,
    "base": Base,
    "lstd": SPD,
    "ssd-7b": Ssd,
    "ssd-13b": Ssd,
}

def main(args):
    if args.infer_type == "lade":
        print("modeling_lade")
        import lade
        import os
        os.environ["LOAD_LADE"]='1'
        os.environ["USE_LADE"]='1'
        lade.augment_all()
        #For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7 
        lade.config_lade(LEVEL=7, WINDOW_SIZE=20, GUESS_SET_SIZE=20, DEBUG=0, POOL_FROM_PROMPT=True, USE_FLASH=True)
        from transformers import LlamaForCausalLM
    elif args.infer_type.startswith("ssd"):
        from baseline.modeling_ssd import SsdLlamaForCausalLM as LlamaForCausalLM
    else:
        from modeling_llama import LlamaForCausalLM

    model_name_or_path = args.model_name_or_path
    name = model_name_or_path.split("/")[-1]
    print(model_name_or_path)

    args.output_dir = args.output_dir + f"/{name}"
    path = Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)

    name = f"{name}_max_{args.max_token_len}_min_{args.min_token_len}_step_{args.step_token_len}"
    if args.algo == 7:
        name = f"cs{args.ca_main}_{args.ca_small}_{name}"
    else:
        name = f"algo{args.algo}_{name}"

    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    except:
        config = LlamaConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    if config.model_type == "llama":
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        config = LlamaConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        try:
            model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                device_map="auto",
                # attn_implementation="eager", # use LlamaAttention to test
                attn_implementation="flash_attention_2", # eagle not support flash attention yet
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_safetensors=True,
            )
            print("try use_safetensors")
        except:
            model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                device_map="auto",
                # attn_implementation="eager", # use LlamaAttention to test
                attn_implementation="flash_attention_2", # eagle not support flash attention yet
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_safetensors=False,
            )
            print("not use_safetensors")
    elif config.model_type == "cohere":
        print("modeling cohere")
        try:
            model = CohereForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                # attn_implementation="eager", # use LlamaAttention to test
                attn_implementation="flash_attention_2", # eagle not support flash attention yet
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_safetensors=True,
            )
        except:
            model = CohereForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                # attn_implementation="eager", # use LlamaAttention to test
                attn_implementation="flash_attention_2", # eagle not support flash attention yet
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_safetensors=False,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            # attn_implementation="eager", # use LlamaAttention to test
            attn_implementation="flash_attention_2", # eagle not support flash attention yet
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_safetensors=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

    # NOTE: add pad_token to use padding
    tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    max_gen_len = args.max_gen_len
    # k
    max_sample = args.max_sample
    name = f"{name}_s{max_sample}"

    ctype = ""
    if args.cache_type == "sink":
        kv_cache_manager = SinkCache(
            start_size=args.start_size, recent_size=args.recent_size
        )
        ctype = f"sink_{args.start_size}_{args.recent_size//1024}k"
    elif args.cache_type == "scache":
        kv_cache_manager = ShortCache(
            start_size=args.start_size, recent_size=args.recent_size
        )
        ctype = f"scache_{args.start_size}_{args.recent_size//1024}k"
    elif args.cache_type == "dyn":
        kv_cache_manager = DynamicCache(
            cache_unit_range=(args.dyn_umin, args.dyn_umax),
            kick=1,
            unit=args.dyn_usize,
            start_size=4,
            slow_up_unum=4,
            threshold=0.5,
        )
        ctype = f"dyn({args.dyn_umin},{args.dyn_umax},{args.dyn_usize})"
    elif args.cache_type == "tcp":
        kv_cache_manager = TcpCache(
            cache_unit_range=(args.dyn_umin, args.dyn_umax),
            unit=args.dyn_usize,
        )
        ctype = f"tcp({args.dyn_umin},{args.dyn_umax},{args.dyn_usize})"
    else:
        raise ValueError(f"Invalid cache_type: {args.cache_type}")

    print("Using cache manager:", ctype)

    name = f"{name}_{ctype}"

    # kv_cache_manager = None
    print(f"max_gen_len: {max_gen_len}")
    print(f"max_sample: {max_sample}")
    print(f"cache_size: {args.start_size + args.recent_size}")
    print(f"kv_cache_manager: {kv_cache_manager}")

    print(f"======== infer type: {args.infer_type} ========")
    if args.infer_type == "lstd":
        model = SPD(model, tokenizer=tokenizer, cache_manager=kv_cache_manager)
    elif args.infer_type == "base":
        model = Base(model, tokenizer=tokenizer)
    elif args.infer_type == "lade":
        model = Lade(model, tokenizer=tokenizer)
    elif args.infer_type.startswith("ssd"):
        model = Ssd(model, tokenizer=tokenizer, model_type=args.infer_type)
    else:
        raise ValueError(f"Invalid infer_type: {args.infer_type}")

    try:
        dataset = load_dataset(args.test_data, split="train")
    except:
        dataset = load_dataset('json', data_files=args.test_data, split=f"train")

    dataset_name = args.test_data.split("/")[-1].split(".")[0]

    print(f"testing {name}, dataset: {dataset_name}")

    max_test_count = args.max_test_count
    cnt = 0

    all_decoding_tps = []
    all_decoding_time = []
    all_acc = []
    all_mem_used = []
    x_data = []

    for sample in dataset:
        if len(x_data) < cnt + 1:
            x_data.append([])
            all_mem_used.append([])
            all_decoding_tps.append([])
            all_decoding_time.append([])
            all_acc.append([])

        input_text = sample[args.feature]
        tokenized = tokenizer(input_text, return_tensors="pt", return_attention_mask=True)
        input_ids = tokenized.input_ids.to(model.device)
        mask = tokenized.attention_mask.to(model.device)

        if input_ids.shape[1] < args.max_token_len:
            continue

        generated_len = 0
        for seqlen in range(args.min_token_len, args.max_token_len + 1, args.step_token_len):
        # for seqlen in [4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 100*1024]:
            if args.max_token_len < seqlen:
                break

            # (bs, seqlen)
            input_token = input_ids[:, :seqlen]
            attention_mask = mask[:, :seqlen]

            # each generate a unit to prevent LLM repeat it's answer
            max_gen_unit = 20
            tokenized_label = input_ids[:, seqlen:seqlen + max_gen_len]
            answer_len = max_gen_len

            print("input token size:", input_token.shape)
            batch_size, token_len = input_token.shape
            end_pos = token_len

            past_key_values = None
            # warmup
            model.generate(
                    input_ids=input_token,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    max_gen_len=args.warmup_step,
                    max_sample=max_sample, algo=1, args=args)

            # renew cache manager after warmup
            if isinstance(model, SPD):
                model.cache_manager.reset()
            

            local_all_decoding_tps = []
            local_all_decoding_time = []
            local_all_acc = []
            local_all_mem_used = []
            local_x_data = []
            for start_pos in range(0, answer_len, max_gen_unit):
                total_time = time.time()
                # generation start

                generation_result = model.generate(
                        input_ids=input_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        max_gen_len=max_gen_len,
                        start_off=max(0, end_pos - max_gen_unit),
                        max_sample=max_sample, algo=args.algo, args=args)
                torch.cuda.synchronize()
                total_time = time.time() - total_time

                past_key_values = generation_result.past_key_values
                generated_ids = generation_result.generated_ids
                decode_time = generation_result.decode_time
                accuracy = generation_result.accuracy
                max_sample_list = generation_result.max_sample_list
                tp_list = generation_result.tp_list
                start_token_list = generation_result.start_token_list

                if args.print:
                    generated_text = (
                        tokenizer.decode(
                            generated_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                            spaces_between_special_tokens=False,
                        )
                        .strip()
                        .split(" ")
                    )

                    print()
                    print(" ".join(generated_text), flush=True)

                if isinstance(generated_ids, list):
                    generated_len = len(generated_ids)
                else:
                    generated_len = generated_ids.shape[1]

                if generated_len <= 1:
                    break

                # trimed kv cache to init state
                past_key_values_trimmed = []
                assert past_key_values
                # TODO: support **LLAMA** kvcache truncte only for now
                # k, v (batch, head, seq, hidden_dim)
                for kv in past_key_values:
                    k, v = kv
                    # NOTE: the indexing is specific for bloom. This won't work for other models
                    # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
                    k = k[:, :, :end_pos, :]
                    v = v[:, :, :end_pos, :]
                    kv_trimmed = (k, v)
                    past_key_values_trimmed.append(kv_trimmed)

                past_key_values = past_key_values_trimmed
                attention_mask = torch.ones(batch_size, end_pos).to(model.device)
                input_token = tokenized_label[0, start_pos:start_pos+max_gen_unit].unsqueeze(0)
                end_pos += max_gen_unit

                # NOTE: second per token
                decode_tokens_per_second = batch_size * generated_len / np.sum(decode_time)

                device = next(model.parameters()).device
                memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

                local_x_data.append(token_len)
                local_all_mem_used.append(memory_used)
                local_all_decoding_tps.append(decode_tokens_per_second)
                local_all_decoding_time.append(np.sum(decode_time))
                local_all_acc.append(np.mean(accuracy))


                print(f"{cnt}/{max_test_count} {start_pos}/{answer_len}: input token_len: {token_len}, generated_len {generated_len},decode_time: {decode_time:.2f}, decode_tps: {decode_tokens_per_second:.2f}, accuracy: {np.mean(accuracy):.2f}, s/iter {total_time:.2f}")

                generated_ids = None
                gc.collect()
                torch.cuda.empty_cache()

                # generation unit done

            # this point generation done
            if generated_len <= 1:
                x_data[cnt] = []
                all_mem_used[cnt] = []
                all_decoding_tps[cnt] = []
                all_decoding_time[cnt] = []
                all_acc[cnt] = []
                break

            x_data[cnt].append(np.mean(local_x_data[cnt]))
            all_mem_used[cnt].append(np.mean(local_all_mem_used[cnt]))
            all_decoding_tps[cnt].append(np.mean(local_all_decoding_tps[cnt]))
            all_decoding_time[cnt].append(np.sum(local_all_decoding_time[cnt]))
            all_acc[cnt].append(np.mean(local_all_acc[cnt]))
            print(">>>>>>>> ", seqlen, all_decoding_tps[cnt])

        if generated_len <= 1:
            continue

        cnt += 1
        if cnt >= max_test_count:
            break

    print(f">>> tested {max_test_count} samples")
    # # NOTE: transpose to plot
    # plot_all_decoding_tps = [list(row) for row in zip(*all_decoding_tps)]
    # plot_all_prefill_tps = [list(row) for row in zip(*all_prefill_tps)]
    # plot_all_decoding_time = [list(row) for row in zip(*all_decoding_time)]
    # plot_all_acc = [list(row) for row in zip(*all_acc)]
    # plot_all_mem_used = [list(row) for row in zip(*all_mem_used)]

    # # plot_ave_all_decoding_time = [np.mean(list(row)) for row in zip(*all_decoding_time)]
    # plot_ave_all_decoding_tps = [np.mean(list(row)) for row in zip(*all_decoding_tps)]
    # plot_ave_all_acc = [np.mean(list(row)) for row in zip(*all_acc)]

    # # draw decoding time graph
    # draw_line_char(plot_all_decoding_tps, x_data=x_data[0],title=f"{name}_tps, total_time={np.sum(all_decoding_time):.2f}", show=False, save_path=f"{args.output_dir}/{args.infer_type}_tp_decode_time_{name}_ds_{dataset_name}.png", filter=False)
    # # draw accuracy graph
    # draw_line_char(plot_all_acc, x_data=x_data[0], title=f"{name}_acc, mean={np.mean(all_acc):.2f}", show=False, save_path=f"{args.output_dir}/{args.infer_type}_tp_accuracy_{name}_ds_{dataset_name}.png", filter=False)
    # # draw memory used graph
    # draw_line_char(plot_all_mem_used, x_data=x_data[0], title=f"{name}_mem", show=False, save_path=f"{args.output_dir}/{args.infer_type}_tp_mem_use_{name}_ds_{dataset_name}.png", filter=False)
    # # draw ave graph
    # draw_line_char(plot_ave_all_decoding_tps, x_data=x_data[0],title=f"{name}_ave_tps, total_time={np.sum(all_decoding_time):.2f}", show=False, save_path=f"{args.output_dir}/{args.infer_type}_tp_ave_decode_time_{name}_ds_{dataset_name}.png", filter=False)
    # draw_line_char(plot_ave_all_acc, x_data=x_data[0], title=f"{name}_ave_acc, mean={np.mean(all_acc):.2f}", show=False, save_path=f"{args.output_dir}/{args.infer_type}_tp_ave_accuracy_{name}_ds_{dataset_name}.png", filter=False)

    # save all raw data as csv
    with open(f"{args.output_dir}/{args.infer_type}_tp_data_{name}_ds_{dataset_name}.csv", "w") as file:
        for i in range(len(x_data)):
            write_csv_line(file, "token_len", x_data[i])
            write_csv_line(file, "decode_tps", all_decoding_tps[i])
            write_csv_line(file, "decode_time", all_decoding_time[i])
            write_csv_line(file, "accuracy", all_acc[i])
            write_csv_line(file, "memory_used", all_mem_used[i])

    # save ave data as csv
    with open(f"{args.output_dir}/{args.infer_type}_AVE_{name}_ds_{dataset_name}.csv", "w") as file:
        write_csv_line(file, "token_len", x_data[0])
        write_csv_line(file, "decode_tps", [np.mean(list(row)) for row in zip(*all_decoding_tps)])
        write_csv_line(file, "decode_time", [np.mean(list(row)) for row in zip(*all_decoding_time)])
        write_csv_line(file, "accuracy", [np.mean(list(row)) for row in zip(*all_acc)])
        write_csv_line(file, "memory_used", [np.mean(list(row)) for row in zip(*all_mem_used)])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--feature", type=str, default="text")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=1024 * 4)
    parser.add_argument("--max_sample", type=int, default=8)
    parser.add_argument("--max_token_len", type=int, default=64 * 1024)
    parser.add_argument("--min_token_len", type=int, default=4 * 1024)
    parser.add_argument("--step_token_len", type=int, default=4 * 1024)
    parser.add_argument("--warmup_step", type=int, default=10)
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--infer_type", type=str, default="lstd", help="lstd, base or eagle")
    # parser.add_argument("--eagle_path", type=str, help="Eagle head path for eagle inference.")
    parser.add_argument("--cache_type", type=str, help="Cache manager type.", default="sink")
    parser.add_argument("--dyn_umax", type=int, help="max unit num.", default=16)
    parser.add_argument("--dyn_umin", type=int, help="min unit num.", default=8)
    parser.add_argument("--dyn_usize", type=int, help="unit size.", default=256)
    parser.add_argument("--max_gen_len", type=int, help="max generation len", default=1024)
    parser.add_argument("--max_test_count", type=int, help="max test sample number", default=10)
    parser.add_argument("--ca_small", type=int, help="small local", default=8)
    parser.add_argument("--ca_main", type=int, help="main local", default=64)
    # TODO: template
    parser.add_argument("--algo", type=int, help="max generation len", default=1)

    args = parser.parse_args()

    main(args)






