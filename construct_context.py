import argparse
from utils.utils import *
def construct_conflict_context(args):
    output_list = []
    units = read_data(args.result_name_tf)
    for i in units:
        A_idx = i[args.input_key_tf].lower().find('A:'.lower())+3
        A_end_idx = i[args.input_key_tf].lower().find('B:'.lower())-1
        B_idx = i[args.input_key_tf].lower().find('B:'.lower())+3
        B_end_idx = i[args.input_key_tf].lower().find('Follow the'.lower())-1
        
        A_answer = i[args.input_key_tf][A_idx:A_end_idx].strip()
        B_answer = i[args.input_key_tf][B_idx:B_end_idx].strip()

        start_index = i[args.input_key_tf].find("Information: ")
        end_index = i[args.input_key_tf].find("\n\nOptions:\nA")
        if i['Y']:
            new_string = i[args.input_key_tf][:start_index] + "Information: " + B_answer + i[args.input_key_tf][end_index:]
            i[args.input_key_tf]=new_string
        else:
            new_string = i[args.input_key_tf][:start_index] + "Information: " + A_answer + i[args.input_key_tf][end_index:]
            i[args.input_key_tf]=new_string
            
    output_path = f"/workspace/wangjiatai/Swin-VIP/data/truthfulQA_label/{args.model_name}_{args.data_name}_all_wrong.jsonl"
    output_list += units
    with jsonlines.open(output_path, mode='w') as writer:
                writer.write_all(output_list)
    


def construct_qwen_single_choice(args):
    output_list = []
    units = read_data(args.result_name_tf)
    def contains_any(a, b):
        return any(s in b for s in a)
    for i in units:
        i['Y'] = contains_any(i['ground_truth'], i['memory_answer'])
        i['counter_memory_answer_ab'] = '[INST] <<SYS>>\nAccording to the given information and your knowledge, choose the best choice from the following options. Follow the format: Option: your option.\n<</SYS>>\n\nInformation: ' + i['counter_memory'] + '\n\nQuestion: ' + i['question'] + '\n\nOptions:\nA: ' + i['memory_answer'] + '\nB: ' + i['counter_answer'] + '\n\n [/INST]'
    output_path = f"/workspace/wangjiatai/Swin-VIP/data/conflictQA_label/conflictQA-popQA-qwen-7b_v1.jsonl"
    output_list += units
    with jsonlines.open(output_path, mode='w') as writer:
                writer.write_all(output_list)  

def test_time(args):
    units = read_data(args.result_name_tf)
    time_list = []
    for i in units:
        time_list.append(i['SwinVIB_time'])
    print(sum(time_list)/len(time_list))
    
      






def constrcut_QA_context(args):
    output_list = []
    units = read_data(args.result_name_tf)
    if args.data_name == 'conflictQA-popQA-llama2-7b':
        for i in units:
            prompt = '[INST] <<SYS>>\nAccording to the information provided and your knowledge, answer the following question.\nUse two sentences maximum and keep the answer as concise as possible.\n'
            query = '<</SYS>>\n\nQuestion: ' + i['question']
            information = '\n\nInformation: ' + i['counter_memory']
            question = '\n\n'+ 'Answer:' +' [/INST]'
            i[args.new_attributes] = prompt + query + information + question
    else:
        for i in units:
            start_index = i[args.input_key_tf].find("Information: ")
            end_index = i[args.input_key_tf].find("\n\nOptions:\nA")
            prompt = '[INST] <<SYS>>\nAccording to the information provided and your knowledge, answer the following question.\nUse two sentences maximum and keep the answer as concise as possible.\n'
            query = '<</SYS>>\n\nQuestion: ' + i['Question']
            information = '\n\nInformation: ' + i[args.input_key_tf][start_index+13:end_index]
            question = '\n\n'+ 'Answer:' +' [/INST]'
            i[args.new_attributes] = prompt + query + information + question
    output_path = f"/workspace/wangjiatai/Swin-VIP/data/truthfulQA_label/{args.model_name}_{args.data_name}_task_QA.jsonl"
    output_list += units
    with jsonlines.open(output_path, mode='w') as writer:
                writer.write_all(output_list)
     

def constrcut_RAG_context(args):
    output_list = []
    units = read_data(args.result_name_tf)
    from sentence_transformers import SentenceTransformer, util
    import torch
    model = SentenceTransformer('/workspace/wangjiatai/Swin-VIP/checkpoint/m3e-base')
    for i in units:
        prompt = '[INST] <<SYS>>\nYou are an assistant for question-answering tasks.\nUse the following pieces of retrieved context  and your own knowledge to answer the question.\nAmong the multiple contexts retrieved, some are correct while others are incorrect.\nUse two sentences maximum and keep the answer concise.\n<</SYS>>\n\n'
        query = 'Question: ' + i['Question']
        context_list = (i['Correct Answers']+'; '+i['Incorrect Answers']).split(';')
        query_embedding = model.encode(i['Question'])
        context_embedding = model.encode(context_list)
        cosine_scores = util.pytorch_cos_sim(query_embedding, context_embedding)
        k = min(5, len(context_list))
        top_results = torch.topk(cosine_scores, k=k)
        top_sentences = []
        for score, idx in zip(top_results[0][0], top_results[1][0]):
            top_sentences.append(context_list[idx])
        context_rag_list = [f"{idx + 1}. {answer.strip()}" for idx, answer in enumerate(top_sentences)]
        context_string = "; ".join(context_rag_list)
        information = '\n\nContext: ' + context_string
        question = '\n\n'+ 'Answer:' +' [/INST]'
        i[args.new_attributes] = prompt + query + information + question
    output_path = f"/workspace/wangjiatai/Swin-VIP/data/truthfulQA_label/{args.model_name}_{args.data_name}_task_RAG.jsonl"
    output_list += units
    with jsonlines.open(output_path, mode='w') as writer:
                writer.write_all(output_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_name_tf', type=str, default='/workspace/wangjiatai/Swin-VIP/runs/Llama_conflictQA-popQA-llama2-7b_withSwinVIP_mu_ab.jsonl')
    parser.add_argument('--input_key_tf', type=str, default='single_evidence_true_false')
    parser.add_argument('--new_attributes', type=str, default='QA', help='QA or RAG')
    parser.add_argument("--data_name", type=str, default='conflictQA-popQA-llama2-7b', help='conflictQA-popQA-llama2-7b or TruthfulQA')
    parser.add_argument("--model_name", type=str, default='qwen2')
    args = parser.parse_args()
    #construct_conflict_context(args)
    #constrcut_QA_context(args)
    #constrcut_RAG_context(args)
    #construct_qwen_single_choice(args)
    test_time(args)