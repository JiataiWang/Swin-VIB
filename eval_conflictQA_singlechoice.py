import argparse
import tqdm
from model.SwinVIB import SwinVIB
from utils.utils import *
from tqdm import tqdm
import numpy as np
import torch
import subprocess
import math
import matplotlib.pyplot as plt

def evalution(args):
    
    def get_cnt(units, right_units, wrong_units, c_units, no_answer_units, input_key, key, answer_ab):
        a_cnt = 0
        b_cnt = 0
        defense_ex_errors = []
        correct_self_errors = []
        mislead = []
        overconfidence = []

        for i in units:   
            A_idx = i[input_key].lower().find('A:'.lower())+3 
            A_end_idx = i[input_key].lower().find('B:'.lower())-1  
            B_idx = i[input_key].lower().find('B:'.lower())+3  
            B_end_idx = i[input_key].lower().find('[/INST]'.lower())-1  
            
            A_answer = i[input_key][A_idx:A_end_idx].strip()  
            B_answer = i[input_key][B_idx:B_end_idx].strip()  

            a_idx = 100000
            for j in ['option: (a)', 'option: a.','option: a ', 'option: a,', 'option a', '(a)', 'a:', 'cannot choose option b', ':a', ': a', A_answer]:
                a = i[key].lower().find(j.lower())   
                a_idx = min(a_idx, a) if a != -1 else a_idx
            b_idx = 100000
            for j in ['option: (b)', 'option: b.','option: b ', 'option: b,', 'option b', '(b)', 'b:', 'cannot choose option a', ':b', ': b', B_answer]:
                b = i[key].lower().find(j.lower())
                b_idx = min(b_idx, b) if b != -1 else b_idx
            c_idx = 100000
            for j in ['option: (c)', 'option: c.','option: c ', 'option: c,', 'option c', '(c)', 'c:', ':c', ': c']:
                c = i[key].lower().find(j.lower())
                c_idx = min(c_idx, c) if c != -1 else c_idx
            if a_idx == 100000 and b_idx == 100000 and c_idx == 100000:  
                no_answer_units.append(i)
            else:
                min_idx = min(a_idx, b_idx, c_idx) 
                if min_idx == a_idx:   
                    if answer_ab and i['Y']: 
                        right_units.append(i)
                        defense_ex_errors.append(i)
                    elif not answer_ab and not i['Y']:  
                        right_units.append(i)
                        correct_self_errors.append(i)
                    else:
                        wrong_units.append(i)  
                        if answer_ab and not i['Y']:
                            overconfidence.append(i)
                        elif not answer_ab and i['Y']:
                            mislead.append(i)
                    a_cnt += 1
                elif min_idx == b_idx:
                    if answer_ab and not i['Y']:  
                        right_units.append(i)
                        correct_self_errors.append(i)
                    elif not answer_ab and i['Y']:  
                        right_units.append(i)
                        defense_ex_errors.append(i)
                    else:
                        wrong_units.append(i)
                        if answer_ab and not i['Y']:
                            overconfidence.append(i)
                        elif not answer_ab and i['Y']:
                            mislead.append(i)
                    b_cnt += 1
                elif min_idx == c_idx:
                    wrong_units.append(i)
        return defense_ex_errors, correct_self_errors, mislead, overconfidence
                    

    def print_cnt(a_units, b_units, c_units, no_answer_units, true_false=True):
        print('------------------------------------------------')
        print('------------------------------------------------')
        if true_false:
            Acc = len(a_units)/(len(a_units)+len(b_units)+len(c_units)+len(no_answer_units))
            print(len(a_units))
        else:
            Acc = len(b_units)/(len(a_units)+len(b_units)+len(c_units)+len(no_answer_units))
            print(len(b_units))
            print('False_True Acc: ', Acc)
        return Acc
    
    def eval_conflict_mi(D, L, L_bar, C_r_ab, C_w_ab, C_na_ab, ab_defense, ab_correct, ab_mislead, ab_overconfidence):
           
            epsilon = 1e-10
            print('------------------------------------------------')
            
            P_L = float(L / D)  
            P_L_bar = L_bar / D       

            
            P_C_r_ab = C_r_ab / D  
            P_C_w_ab = C_w_ab / D  
            P_C_na_ab = C_na_ab / D  

            if not np.isclose(C_r_ab + C_w_ab + C_na_ab, D):
                print("Error: Probabilities do not sum up to 1.")

            import random

            H = - (P_L * math.log2(P_L + epsilon) + P_L_bar * math.log2(P_L_bar + epsilon))
            print('no_add_system_total_information: ', H)
            
            # 计算引入冲突条件熵 H(A|B)
            H_A_given_B = - (P_C_r_ab * math.log2(P_C_r_ab + epsilon) + 
                            P_C_w_ab * math.log2(P_C_w_ab + epsilon) + 
                            P_C_na_ab * math.log2(P_C_na_ab + epsilon))
            print('TRE: ', H_A_given_B)
            

            P_C_correct = ab_correct / L_bar 
            P_C_defense = ab_defense / L
            

            P_L_bar_correct = (L_bar - ab_correct) / L_bar 
            P_L_defense = (L - ab_defense) / L



            H_correct = - (P_C_correct * math.log2(P_C_correct + epsilon) + P_L_bar_correct * math.log2(P_L_bar_correct + epsilon))
            H_defense = - (P_C_defense * math.log2(P_C_defense + epsilon) + P_L_defense * math.log2(P_L_defense + epsilon))

    

            print('ACC: ', P_C_r_ab)
            print('------------------------------------------------')
            print('Correction_Rate: ', P_C_correct)
            print('Resistance_Rate: ', P_C_defense)
            print('------------------------------------------------')

 
    
    #AB选择题
    ab_units = read_data(args.result_name_tf)
    answer_ab = 'answer_ab' in args.output_key_tf
    ab_right_units = []
    ab_wrong_units = []
    ab_c_units = []
    ab_no_answer_units = []
    defense_ex_errors, correct_self_errors, mislead, overconfidence = get_cnt(ab_units, ab_right_units, ab_wrong_units, ab_c_units, ab_no_answer_units, args.input_key_tf, args.output_key_tf, answer_ab)
    ab_acc = print_cnt(ab_right_units, ab_wrong_units, ab_c_units, ab_no_answer_units, answer_ab)
    D = len(ab_units)
    L = sum(1 for item in ab_units if item['Y'] == True)
    L_bar = D-L
    C_r_ab = len(ab_right_units)
    C_w_ab = len(ab_wrong_units)
    C_na_ab = len(ab_no_answer_units)
    defense = len(defense_ex_errors)
    correct = len(correct_self_errors)
    mislead =len(mislead)
    overconfidence = len(overconfidence)
    eval_conflict_mi(D, L, L_bar, C_r_ab, C_w_ab, C_na_ab, defense, correct, mislead, overconfidence)
    print(args.result_name_tf)
    

    




    


    
    




    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_name_tf', type=str, default='/Users/titus.w/Documents/博士生阶段/课题/第五个课题/AAAI投稿/code/Swin-VIB/runs/runs/Llama-2-7b-chat-hf_conflictQA-popQA-llama2-7b_VIB_mu_ab.jsonl')
    parser.add_argument('--input_key_tf', type=str, default='counter_memory_answer_ab')
    parser.add_argument('--output_key_tf', type=str, default='counter_memory_answer_ab_output')
    args = parser.parse_args()
    evalution(args)
    
