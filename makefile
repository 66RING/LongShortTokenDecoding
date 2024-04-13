run:
	# python ./main.py --model_name_or_path $(MODELS_DIR)/tinyllama-110M
	python ./main.py --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf
	# python ./main.py --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf

eval:
	# python ./eval.py  --model_name_or_path $(MODELS_DIR)/Yarn-Llama-2-7b-128k --test_data $(DATASETS_DIR)/128kset/test_books3_129_128k-1000k.jsonl
	python ./eval.py  --model_name_or_path $(MODELS_DIR)/tinyllama-110M --test_data $(DATASETS_DIR)/128kset/test_books3_129_128k-1000k.jsonl

yarn:
	# python ./main.py --model_name_or_path $(MODELS_DIR)/LLaMA-7B-PoSE-YaRN-128k
	python ./main.py --model_name_or_path $(MODELS_DIR)/Yarn-Llama-2-7b-128k

stream:
	python ./run_streaming_llm.py --model_name_or_path $(MODELS_DIR)/tinyllama-110M

7b:
	# python ./run_streaming_llm.py --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf
	python ./run_streaming_llm.py --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf

gen_test:
	python ./tools/gen_test_data.py --file_path ./ignore/harry_potter_all.txt --output ./test_data.jsonl

# test test throughput
tp:
	# python ./eval_throughput.py  --model_name_or_path $(MODELS_DIR)/tinyllama-110M --test_data ./test_data.jsonl
	python ./eval_throughput.py  --model_name_or_path $(MODELS_DIR)/Yarn-Llama-2-7b-128k --test_data ./test_data.jsonl
	# python ./eval_throughput.py  --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf --test_data ./test_data.jsonl

df data_filter:
	# python ./tools/data_filter.py --min_token_len 131072 --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf --output test_books3_129_128k-1000k.jsonl --dataset $(DATASETS_DIR)/PoSE-Datasets/pile/test_books3.jsonl
	# python ./tools/data_filter.py --min_token_len 131072 --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf --output test_pg19_20_130k-360k.jsonl --dataset $(DATASETS_DIR)/PoSE-Datasets/pile/test_pg19.jsonl
	# python ./tools/data_filter.py --min_token_len 131072 --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf --output val_long_46_128k-700k.jsonl --dataset $(DATASETS_DIR)/PoSE-Datasets/pile/val_long.jsonl
	# python ./tools/data_filter.py --min_token_len 131072 --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf --output proof_pile_36_128k-560k.jsonl --dataset $(DATASETS_DIR)/PoSE-Datasets/proof-pile/test.jsonl
	# python ./tools/data_filter.py --min_token_len 131072 --model_name_or_path $(MODELS_DIR)/Llama-2-7b-chat-hf --output gov_report_3_140k-300k.jsonl --dataset $(DATASETS_DIR)/PoSE-Datasets/scrolls/gov_report/test_long.jsonl --feature input

