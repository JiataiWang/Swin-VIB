from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2-7B-Instruct',cache_dir='/workspace/wangjiatai/')
print(model_dir) 