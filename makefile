run:
	python ./main.py --model_name_or_path /home/ring/Documents/workspace/modules/tinyllama-110M

stream:
	python ./run_streaming_llm.py --model_name_or_path /home/ring/Documents/workspace/modules/tinyllama-110M

7b:
	python ./run_streaming_llm.py --model_name_or_path /home/ring/Documents/workspace/modules/Llama-2-7b-chat-hf
	# python ./run_streaming_llm.py --model_name_or_path /raid/ljl66ring/var/models/Llama-2-7b-chat-hf


