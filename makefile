run:
	python ./main.py --model_name_or_path /home/ring/Documents/workspace/modules/tinyllama-110M
	# python ./main.py --model_name_or_path /raid/ljl66ring/var/models/Llama-2-7b-chat-hf

yarn:
	# python ./main.py --model_name_or_path /home/ring/Documents/workspace/modules/LLaMA-7B-PoSE-YaRN-128k
	python ./main.py --model_name_or_path /raid/ljl66ring/DeepSpeed-Chat/training/step1_supervised_finetuning/train_with_pgbook/yarn-pose-4096win-32scal-151sstep_150

stream:
	python ./run_streaming_llm.py --model_name_or_path /home/ring/Documents/workspace/modules/tinyllama-110M

7b:
	# python ./run_streaming_llm.py --model_name_or_path /home/ring/Documents/workspace/modules/Llama-2-7b-chat-hf
	python ./run_streaming_llm.py --model_name_or_path /raid/ljl66ring/var/models/Llama-2-7b-chat-hf

gen_test:
	python ./tools/gen_test_data.py --file_path ./ignore/harry_potter_all.txt --output ./test_data.jsonl

