# llama2-gpu
Running LLAMA2 on GPU and creating sentence embeddings

## Usage
1. Get access to LLAMA2 weights on [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) page.
2. Get the huggingface token 
3. Create model shards using [`make_shards.py`](make_shards.py)
4. Run [`main.py`](main.py) to generate LLAMA2 sentence embeddings for a batch of sentences 