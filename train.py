import subprocess
def query_gpu_memory():
        smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader']).decode('utf-8')
        gpu_memory_free = {}
        for line in smi_output.strip().split('\n'):
            index, free_memory = line.split(',')
            gpu_memory_free[int(index)] = float(free_memory.strip())
        return gpu_memory_free
gpu_memory_free = query_gpu_memory()
best_gpu = max(gpu_memory_free, key=gpu_memory_free.get)
print(f"Selected GPU {best_gpu} with {gpu_memory_free[best_gpu]} MB of free memory.")

import argparse
from model.SwinVIB import SwinVIB
from utils.utils import *
import torch


def extract(args):
    print(args)
    def query_gpu_memory():
        # 调用 nvidia-smi 来获取 GPU 信息
        smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader']).decode('utf-8')
        gpu_memory_free = {}
        # 解析输出
        for line in smi_output.strip().split('\n'):
            index, free_memory = line.split(',')
            gpu_memory_free[int(index)] = float(free_memory.strip())
        return gpu_memory_free
    gpu_memory_free = query_gpu_memory()
    best_gpu = max(gpu_memory_free, key=gpu_memory_free.get)
    print(f"Selected GPU {best_gpu} with {gpu_memory_free[best_gpu]} MB of free memory.")
    args.device = torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {args.device}")
    data = read_data(args.data_path)
    ic = SwinVIB(args)
    if args.data_name == 'TruthfulQA':
        ic.ex_truthfulqa(data)
    elif args.data_name == 'conflictQA-popQA-llama2-7b' or 'conflictQA-popQA-qwen-7b':
        ic.ex_conflictqa(data)
    #ic.plot_attention(data, args.data_name)
    #ic.plot_mse(data, args.data_name)

    

def trian(args):
    print(args)
    def query_gpu_memory():
        # 调用 nvidia-smi 来获取 GPU 信息
        smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader']).decode('utf-8')
        gpu_memory_free = {}
        # 解析输出
        for line in smi_output.strip().split('\n'):
            index, free_memory = line.split(',')
            gpu_memory_free[int(index)] = float(free_memory.strip())
        return gpu_memory_free
    gpu_memory_free = query_gpu_memory()
    best_gpu = max(gpu_memory_free, key=gpu_memory_free.get)
    print(f"Selected GPU {best_gpu} with {gpu_memory_free[best_gpu]} MB of free memory.")
    args.device = torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {args.device}")
    ic = SwinVIB(args)
    ic.trian()
    #ic.tr(data)

def infer(args):
    print(args)
    args.device = torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {args.device}")
    data = read_data(args.data_path)
    ic = SwinVIB(args)
    #ic.infer(data)
    ic.infer(data)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama')
    parser.add_argument("--model_path", type=str, default='/workspace/wangjiatai/Llama/Llama-2-7b-chat-hf')
    parser.add_argument("--data_name", type=str, default='conflictQA-popQA-llama2-7b', help='conflictQA-popQA-llama2-7b or TruthfulQA')
    parser.add_argument("--data_path", type=str, default='/workspace/wangjiatai/Swin-VIP/data/conflictQA_label/conflictQA-popQA-llama2-7b.jsonl')
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--input_key", type=str, default='counter_memory_answer_ab', help='counter_memory_answer_ab or single_evidence_true_false or QA or RAG')
    parser.add_argument("--output_key", type=str, default='counter_memory_answer_ab_output', help='counter_memory_answer_ab_output or single_evidence_true_false_output or QA_output or RAG_output')
    parser.add_argument("--data_generate_path", type=str, default='/workspace/wangjiatai/Swin-VIP/data/conflictQA_label/conflict_qwen2_dataset.pkl')
    parser.add_argument("--output_path", type=str, default='/workspace/wangjiatai/Swin-VIP/runs')
    parser.add_argument("--output_pic_path", type=str, default='/workspace/wangjiatai/Swin-VIP/runs/')
    parser.add_argument("--is_SwinVIP", type=bool, default=True)
    parser.add_argument("--task", type=str, default='single_choice', help='single_choice or QA or RAG')
    parser.add_argument("--mode", type=str, default='montecarlo', help='mu or montecarlo')
    parser.add_argument("--checkpoint", type=str, default='/workspace/wangjiatai/Swin-VIP/checkpoint/conflictqa-llama2', help='conflictQA-popQA-llama2-7b or TruthfulQA')
    args = parser.parse_args()
    #extract(args)
    #trian(args)
    infer(args)
    