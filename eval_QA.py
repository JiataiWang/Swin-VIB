import argparse
from utils.utils import *
import math
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import CHRF
# from datasets import Dataset
import pandas as pd
from tqdm import tqdm
#from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, AutoModelForMaskedLM
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


def calculate_entropy(text):
    count = Counter(text)
    total = len(text)
    entropy = -sum((freq / total) * math.log2(freq / total) for freq in count.values())
    return entropy

def evalution(args):

    total_units = read_data(args.result_name_tf)
    results_df = pd.DataFrame()

    for i in tqdm(total_units, desc="Processing Units"):  
        candidate_text = i[args.key_out]
        reference_text = i[args.groudtruth]
        reference = reference_text.split()
        candidate = candidate_text.split()

        entropy1 = calculate_entropy(candidate) 
        entropy2 = calculate_entropy(reference)  

        # BLEU-4
        bleu_score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
        print("BLEU-4 score:", bleu_score)

        # METEOR
        meteor = meteor_score([reference], candidate)
        print("METEOR score:", meteor)

        # ChrF
        chrf = CHRF().corpus_score([candidate_text], [[reference_text]]).score
        print("ChrF score:", chrf)

        # COMET
        #comet_score = comet_model.predict([{"src": reference_text, "mt": candidate_text, "ref": reference_text}])
        #print("COMET score:", comet_score['predicted_scores'][0])

        current_result = pd.DataFrame([{
            "entropy1": entropy1,
            "entropy2": entropy2,
            "bleu_score": bleu_score,
            "meteor": meteor,
            "chrf": chrf,
            #"comet_score": comet_score['predicted_scores'][0],
        }])
        results_df = pd.concat([results_df, current_result], ignore_index=True)

    results_df.to_excel(f'{args.result_name_tf[:-6]}_{args.key}_.xlsx', index=False)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_name_tf', type=str, default='/workspace/wangjiatai/Swin-VIP/runs/Qwen_conflictQA-popQA-qwen-7b_noSwinVIP_mu_QA.jsonl')
    parser.add_argument('--key', type=str, default='QA', help='QA or RAG')
    parser.add_argument('--key_out', type=str, default='QA_output', help='QA_output or RAG_output')
    parser.add_argument('--groudtruth', type=str, default='counter_memory', help='Best Answer or counter_memory')
    args = parser.parse_args()
    evalution(args)