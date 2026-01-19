import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import jsonlines
from utils.plot import *
from utils.utils import *
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
def preprocess_input(x):
    nan_mask = torch.isnan(x)
    inf_mask = torch.isinf(x)
    abnormal_mask = nan_mask | inf_mask

    if abnormal_mask.any():
        normal_values = x[~abnormal_mask]
        if normal_values.numel() > 0:
            median_value = normal_values.median()
        else:
            median_value = torch.tensor(0.0, device=x.device)
        x[abnormal_mask] = median_value
    return x

def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
class CustomDataset(Dataset):
    def __init__(self, data):
        # 初始化数据
        self.data = data

    def __len__(self):
        # 返回数据集中的元素数量
        return len(self.data)

    def __getitem__(self, idx):
        # 获取第 idx 个元组，其中包含 ndarry 和 标签
        input_data, label = self.data[idx]
        # 转换成 torch.Tensor，这里假设输入数据是 numpy ndarray
        input_tensor = torch.from_numpy(input_data).float()
        label_tensor = torch.tensor(label).float()
        return input_tensor, label_tensor


#llama-conflict 
class QDistributionVIB(nn.Module):
    def __init__(self, beta=0.0001):
        super(QDistributionVIB, self).__init__()
        self.beta = beta
        self.relu = nn.ReLU()
        #encoder
        self.fc1 = nn.Linear(4096, 1024)
        self.fc_mu = nn.Linear(1024, 128)
        self.fc_log_var = nn.Linear(1024, 128)
        
        #decoder
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        #encoder
        x = self.fc1(x)
        mu = self.fc_mu(x)
        logvar = self.fc_log_var(x)
        z = self.reparameterize(mu, logvar)

        #decoder
        z = self.relu(self.fc2(z))
        y_pred = self.fc3(z)
        y_pred = self.sigmoid(y_pred)
        y_pred = y_pred.squeeze(-1)
        return y_pred, mu, logvar

    def loss_function(self, pred, true_y, mu, logvar):
        bce_loss = F.binary_cross_entropy(pred, true_y, reduction='sum')
        kl_loss = -0.5 * torch.sum(torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        return bce_loss , kl_loss
    
    def detech_mu(self, x):
        x = self.fc1(x)
        mu = self.fc_mu(x)
        z = self.relu(self.fc2(mu))
        y_pred = self.fc3(z)
        y_pred = self.sigmoid(y_pred)
        y_pred = y_pred.squeeze(-1)
        return y_pred

class SwinVIB:
    def __init__(self,
                 args):
        import torch.fx
        import model.llama as llama
        import model.qwen2 as qwen2
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, legacy=False)
        self.model = llama.LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map='auto')
        #self.model = qwen2.Qwen2ForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map='auto')
        self.tokenizer.pad_token_id = 0     
        self.tokenizer.padding_side = 'left'
        self.batch_size = args.batch_size
        self.input_key = args.input_key
        self.output_key = args.output_key
        self.model_name = args.model_name
        self.data_name = args.data_name
        self.window = 7
        self.max_new_tokens = 512
        # if args.model_name == 'Llama-2-7b-chat-hf':
        #     #self.qdistribution = QDistribution()
        #     self.qdistribution = QDistributionVIB()
        self.output_path = args.output_path
        self.output_pic_path = args.output_pic_path
        self.data_generate_path = args.data_generate_path
        self.mode = args.mode
        self.task = args.task
        self.is_SwinVIP = args.is_SwinVIP
   
    def generate(self, inputs, max_new_tokens=256):
        outputs = self.model.generate(input_ids=inputs.input_ids.cuda(0), attention_mask=inputs.attention_mask.cuda(0), 
                    max_new_tokens=max_new_tokens, 
                    return_dict_in_generate=True, 
                    repetition_penalty=1, 
                    temperature=1.0,
                    top_p=1.0,
                    do_sample  = False
                        )
        return outputs
     
    def ex_conflictqa(self, data):
        R = [[] for _ in range(len(self.model.model.layers))]
        y = [0 if item['Y'] else 1 for item in data]  
        for id in tqdm(range(0, len(data), self.batch_size)): 
            encodings = self.tokenizer([i[self.input_key] for i in data[id:min(id+self.batch_size,len(data))]], return_tensors='pt', padding=True)
            y_batch = y[id:min(id+self.batch_size,len(y))]
            with torch.no_grad():
                _ = self.model(**encodings)
            tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
            start_id = [i.index('Information')+2 for i in tokenized]
            end_id = [i.index('Question') for i in tokenized]
            encodings = encodings.to(self.device)
            for i in range(len(self.model.model.layers)):
                attn_score = self.model.model.layers[i].self_attn.attn_score      #llama(24, 304, 4096) ##qwen(24, 259, 3584)
                for idx in range(len(start_id)):
                    head_score = attn_score[idx,start_id[idx]:end_id[idx]]
                    random_head_score = random_select(head_score)
                    random_head_score = np.mean(random_head_score, axis=0)
                    R[i].append((random_head_score, y_batch[idx]))
        with open(self.data_generate_path, 'wb') as f:
            pickle.dump(R, f)

    def ex_truthfulqa(self, data):
        R = [[] for _ in range(len(self.model.model.layers))]
        y = [1 if item['single_evidence_true_false_label'] else 0 for item in data] 
        for id in tqdm(range(0, len(data), self.batch_size)):
            encodings = self.tokenizer([i[self.input_key] for i in data[id:min(id+self.batch_size,len(data))]], return_tensors='pt', padding=True)
            y_batch = y[id:min(id+self.batch_size,len(y))]
            with torch.no_grad():
                _ = self.model(**encodings)
            tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
            start_id = []
            for i in range(len(tokenized)):
                for j in range(len(tokenized[i])):
                    if tokenized[i][j] == 'Information':
                        if tokenized[i][j+1] == ':':
                            start_id.append(j+1)
                            break
            end_id = [i.index('Options')-3 for i in tokenized]
            encodings = encodings.to(self.device)
            for i in range(len(self.model.model.layers)):
                attn_score = self.model.model.layers[i].self_attn.attn_score      #(24, 140, 4096)
                for idx in range(len(start_id)):
                    head_score = attn_score[idx,start_id[idx]+1:end_id[idx]+1]
                    random_head_score = random_select(head_score)
                    random_head_score = np.mean(random_head_score, axis=0)
                    R[i].append((random_head_score, y_batch[idx]))
        with open(self.data_generate_path, 'wb') as f:
            pickle.dump(R, f)
    
    def trian(self):
        with open(self.data_generate_path, 'rb') as f:
            R_loaded = pickle.load(f)
        import torch.optim as optim
        num_epochs = 200
        # 使用示例
        directory_path = f'/workspace/wangjiatai/Swin-VIP/checkpoint/qwen_truthful'  # 替换为你的目录路径
        create_directory_if_not_exists(directory_path)
        for i in range(len(self.model.model.layers)): 
            trainset = R_loaded[i]
            dataset = CustomDataset(trainset)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            Q = QDistributionVIB().to(self.device)
            Q.apply(init_weights)
            optimizer = optim.Adam(Q.parameters(), lr=0.0001)
            import matplotlib.pyplot as plt
            epoch_losses = []
            epoch_bces = []
            epoch_kls = []
            for epoch in range(num_epochs):
                total_loss = 0
                total_bce = 0
                total_kl = 0
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    y_pred, mu, logvar = Q(inputs)
                    bce_loss, kl_loss = Q.loss_function(y_pred, labels, mu, logvar)
                    loss = bce_loss + Q.beta*kl_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_bce += bce_loss.item()
                    total_kl +=  kl_loss.item()
                epoch_losses.append(total_loss / len(dataloader))
                epoch_bces.append(total_bce / len(dataloader))
                epoch_kls.append(total_kl / len(dataloader))
                print(f'Layer {i} Beta {Q.beta} || Epoch {epoch+1}/{num_epochs}, Loss: {epoch_losses[-1]}, BCE Loss: {epoch_bces[-1]}, KL Loss: {epoch_kls[-1]}')
            torch.save(Q.state_dict(), f'{directory_path}/{Q.beta}_Swin_VIP_layer_{i}.pth')
            # 绘制损失图
            plt.figure(figsize=(12, 8))
            plt.subplot(311)
            plt.plot(epoch_losses, label='Total Loss')
            plt.title('Total Loss over Epochs')
            plt.legend()

            plt.subplot(312)
            plt.plot(epoch_bces, label='BCE Loss')
            plt.title('BCE Loss over Epochs')
            plt.legend()

            plt.subplot(313)
            plt.plot(epoch_kls, label='KL Loss')
            plt.title('KL Loss over Epochs')
            plt.legend()

            plt.tight_layout()

            # 保存图像到指定文件
            plt.savefig(f'/workspace/wangjiatai/Swin-VIP/runs/pic/{Q.beta}_loss_plot_qwen2_layer_{i}.png')
            plt.show()

    def infer(self, data):
        if self.is_SwinVIP == True and self.data_name == 'conflictQA-popQA-llama2-7b' and self.task == 'single_choice':
            if self.mode == 'mu':
                pass
            elif self.mode == 'montecarlo':
                montecarlo_samples = 10
            print('-------------------------------------')
            models = []
            output_list = []
            time_lits = []
            for i in range(len(self.model.model.layers)):  
                model = QDistributionVIB().to(self.device)
                weight_path = f'/workspace/wangjiatai/Swin-VIP/checkpoint/conflictqa-llama2/1e-06_VIB_layer_{i}.pth'  # 构造权重文件的路径
                model.load_state_dict(torch.load(weight_path, map_location='cuda:0'))
                models.append(model.eval())

            for id in tqdm(range(0, len(data), self.batch_size)): 
                encodings = self.tokenizer([i[self.input_key] for i in data[id:min(id+self.batch_size,len(data))]], return_tensors='pt', padding=True)
                tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
                start_id = [i.index('Information')+2 for i in tokenized]
                end_id = [i.index('Question')-2 for i in tokenized]  
                encodings = encodings.to(self.device)  
                with torch.no_grad():
                    _ = self.model(**encodings)
                result = [torch.ones(end_id[i]-start_id[i]) for i in range(len(start_id))]
                for layer in range(len(self.model.model.layers)):
                    attn_score = self.model.model.layers[layer].self_attn.attn_score
                    for idx in range(len(start_id)):
                        head_score = attn_score[idx,start_id[idx]:end_id[idx]] 
                        j = 0
                        elapsed_time_list = []
                        while j < head_score.shape[0]: 
                            if self.mode=='mu':
                                start_time = time.time()
                                predict_result = models[layer].detech_mu(torch.mean(torch.Tensor(head_score[j:min(j+self.window, head_score.shape[0])]), axis=0).to(self.device))
                                end_time = time.time()
                            elif self.mode=='montecarlo':
                                predictions = []
                                start_time = time.time()
                                for _ in range(montecarlo_samples):
                                    y_pred, _, _ = models[layer](torch.mean(torch.Tensor(head_score[j:min(j+self.window, head_score.shape[0])]), axis=0).to(self.device))
                                    predictions.append(y_pred)
                                end_time = time.time()
                                predictions = torch.stack(predictions)
                                predict_result = predictions.mean(0)
                            for step in range(0, min(self.window, head_score.shape[0]-j)):
                                result[idx][j+step] = result[idx][j+step]+predict_result
                            j = j+min(self.window, head_score.shape[0]-j)
                            elapsed_time = end_time - start_time
                            elapsed_time_list.append(elapsed_time)
                        sigle_sample_time = sum(elapsed_time_list) #这句话一共用了多少时间
                        time_lits.append(sigle_sample_time)
            

                #qwen
                result = [torch.where(item / len(self.model.model.layers) > 0.8, torch.tensor(1.0), torch.tensor(0.0)) for item in result]
                for idx in range(len(result)):
                    encodings.attention_mask[idx][start_id[idx]:end_id[idx]] = result[idx]
                
                start_time_llm = time.time()
                outputs = self.model.generate(input_ids=encodings.input_ids, attention_mask=encodings.attention_mask, 
                        max_new_tokens=self.max_new_tokens, 
                        return_dict_in_generate=True, 
                        repetition_penalty=1, 
                        temperature=1.0,
                        top_p=1.0,
                        do_sample  = False
                            )
                sequences = self.tokenizer.batch_decode(outputs.sequences[:,encodings.input_ids.shape[-1]:], 
                                                            skip_special_tokens=True)
                end_time_llm = time.time()
                
                elapsed_time_llm = (end_time_llm - start_time_llm)/self.batch_size
                
                for i in range(len(sequences)):
                    print(data[id+i][self.input_key])
                    print(sequences[i])
                    print("======")
                    data[id+i][self.output_key] = sequences[i]
                    #data[id+i]['time'] = time_lits[i]
                    data[id+i]['LLM_time'] = elapsed_time_llm
                    data[id+i]['SwinVIB_time'] = sum(time_lits)/self.batch_size
                time_lits.clear()
            output_list += data
            outpath = f"{self.output_path}/Llama_{self.data_name}_withSwinVIP_{self.mode}_{self.input_key[-2:]}.jsonl"
            #self.output_path = f"{self.output_path}/{self.model_name}_{'no_add'}.jsonl"
            with jsonlines.open(outpath, mode='w') as writer:
                writer.write_all(output_list)
        
        if self.is_SwinVIP == True and self.data_name == 'TruthfulQA':
            if self.mode == 'mu':
                pass
            elif self.mode == 'montecarlo':
                montecarlo_samples = 10
            print('-------------------------------------')
            models = []
            output_list = []
            time_lits = []
            for i in range(len(self.model.model.layers)):  
                model = QDistributionVIB().to(self.device)
                weight_path = f'/workspace/wangjiatai/Swin-VIP/checkpoint/qwen_truthful/0.0001_Swin_VIP_layer_{i}.pth'  # 构造权重文件的路径
                model.load_state_dict(torch.load(weight_path, map_location='cuda:0', weights_only=True))
                models.append(model.eval())
            for id in tqdm(range(0, len(data), self.batch_size)):
                encodings = self.tokenizer([i[self.input_key] for i in data[id:min(id+self.batch_size,len(data))]], return_tensors='pt', padding=True)
                tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
                if self.task == 'single_choice':
                    start_id = []
                    for i in range(len(tokenized)):
                        for j in range(len(tokenized[i])):
                            if tokenized[i][j] == 'Information':
                                if tokenized[i][j+1] == ':':
                                    start_id.append(j+2)
                                    break
                    end_id = [i.index('Options') for i in tokenized] #每次都得检查
                elif self.task == 'QA':
                    start_id = [i.index('Information')+2 for i in tokenized]
                    end_id = [i.index('Answer')-2 for i in tokenized]
                elif self.task == 'RAG':
                    start_id = [i.index('Context')+2 for i in tokenized]
                    end_id = [i.index('Answer')-2 for i in tokenized]
                encodings = encodings.to(self.device)  
                with torch.no_grad():
                    _ = self.model(**encodings)
                result = [torch.zeros(end_id[i]-start_id[i]) for i in range(len(start_id))]
                for layer in range(len(self.model.model.layers)):
                    attn_score = self.model.model.layers[layer].self_attn.attn_score
                    for idx in range(len(start_id)):
                        head_score = attn_score[idx,start_id[idx]:end_id[idx]] #一句话一句话来 
                        j = 0
                        elapsed_time_list = []
                        while j < head_score.shape[0]:  #这句话中每个窗口每个窗口来
                            if self.mode=='mu':
                                start_time = time.time()
                                predict_result = models[layer].detech_mu(torch.mean(torch.Tensor(head_score[j:min(j+self.window, head_score.shape[0])]), axis=0).to(self.device))
                                end_time = time.time()
                            elif self.mode=='montecarlo':
                                predictions = []
                                start_time = time.time()
                                for _ in range(montecarlo_samples):
                                    y_pred, _, _ = models[layer](torch.mean(torch.Tensor(head_score[j:min(j+self.window, head_score.shape[0])]), axis=0).to(self.device))
                                    predictions.append(y_pred)
                                end_time = time.time()
                                predictions = torch.stack(predictions)
                                predict_result = predictions.mean(0)
                            for step in range(0, min(self.window, head_score.shape[0]-j)):
                                result[idx][j+step] = result[idx][j+step]+predict_result
                            j = j+min(self.window, head_score.shape[0]-j)
                            elapsed_time = end_time - start_time
                            elapsed_time_list.append(elapsed_time)
                        sigle_sample_time = sum(elapsed_time_list) #这句话一共用了多少时间
                        time_lits.append(sigle_sample_time)

                result = [torch.where(item / len(self.model.model.layers) > 0.6, torch.tensor(1.0), torch.tensor(0.0)) for item in result] 
                for idx in range(len(result)):
                    encodings.attention_mask[idx][start_id[idx]:end_id[idx]] = result[idx]
                
                outputs = self.model.generate(input_ids=encodings.input_ids, attention_mask=encodings.attention_mask, 
                        max_new_tokens=self.max_new_tokens, 
                        return_dict_in_generate=True, 
                        repetition_penalty=1, 
                        temperature=1.0,
                        top_p=1.0,
                        do_sample  = False
                            )
                sequences = self.tokenizer.batch_decode(outputs.sequences[:,encodings.input_ids.shape[-1]:], 
                                                            skip_special_tokens=True)
                
                for i in range(len(sequences)):
                    print(data[id+i][self.input_key])
                    print(sequences[i])
                    print("======")
                    data[id+i][self.output_key] = sequences[i]
                    data[id+i]['time'] = time_lits[i]

            output_list += data
            outpath = f"/workspace/wangjiatai/Swin-VIP/runs_qwen/Qwen_{self.data_name}_{self.task}_with_{self.mode}_SwinVIP.jsonl"
            #self.output_path = f"{self.output_path}/{self.model_name}_{'no_add'}.jsonl"
            with jsonlines.open(outpath, mode='w') as writer:
                writer.write_all(output_list)


        if self.is_SwinVIP == False and self.data_name == 'TruthfulQA':
            is_context = True 
            output_list = []
            for id in tqdm(range(0, len(data), self.batch_size)): #len(data)
                encodings = self.tokenizer([i[self.input_key] for i in data[id:min(id+self.batch_size,len(data))]], return_tensors='pt', padding=True)
                tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
                encodings = encodings.to(self.device)  
                if self.task == 'single_choice':
                    start_id_sc = [i.index('Information')+2 for i in tokenized]
                    end_id_sc = [i.index('Options') for i in tokenized]
                    if is_context == False:
                        result = [torch.zeros(end_id_sc[i]-start_id_sc[i]) for i in range(len(start_id_sc))]
                    else:
                        result = [torch.ones(end_id_sc[i]-start_id_sc[i]) for i in range(len(start_id_sc))]
                    for idx in range(len(result)):
                        encodings.attention_mask[idx][start_id_sc[idx]:end_id_sc[idx]] = result[idx]
                if self.task == 'QA':
                    start_id_qa = [i.index('Information')+2 for i in tokenized]
                    end_id_qa = [i.index('Answer')-2 for i in tokenized]
                    if is_context == False:
                        result = [torch.zeros(end_id_qa[i]-start_id_qa[i]) for i in range(len(start_id_qa))]
                    else:
                        result = [torch.ones(end_id_qa[i]-start_id_qa[i]) for i in range(len(start_id_qa))]
                    for idx in range(len(result)):
                        encodings.attention_mask[idx][start_id_qa[idx]:end_id_qa[idx]] = result[idx]
                if self.task == 'RAG':
                    start_id_rag = [i.index('Context')+2 for i in tokenized]
                    end_id_rag = [i.index('Answer')-2 for i in tokenized]
                    if is_context == False:
                        result = [torch.zeros(end_id_rag[i]-start_id_rag[i]) for i in range(len(start_id_rag))]
                    else:
                        result = [torch.ones(end_id_rag[i]-start_id_rag[i]) for i in range(len(start_id_rag))]
                    for idx in range(len(result)):
                        encodings.attention_mask[idx][start_id_rag[idx]:end_id_rag[idx]] = result[idx]
                outputs = self.model.generate(input_ids=encodings.input_ids, attention_mask=encodings.attention_mask, 
                        max_new_tokens=self.max_new_tokens, 
                        return_dict_in_generate=True, 
                        repetition_penalty=1, 
                        temperature=1.0,
                        top_p=1.0,
                        do_sample  = False
                            )
                sequences = self.tokenizer.batch_decode(outputs.sequences[:,encodings.input_ids.shape[-1]:], 
                                                            skip_special_tokens=True)
                if is_context == False:
                    for i in range(len(sequences)):
                        print(data[id+i][self.input_key])
                        print(sequences[i])
                        print("======")
                        if self.task == 'single_choice':
                            data[id+i]['orignal_answer'] = sequences[i]
                            A_idx = data[id+i][self.input_key].lower().find('A:'.lower())+3
                            A_end_idx = data[id+i][self.input_key].lower().find('B:'.lower())-1
                            A_answer = data[id+i][self.input_key][A_idx:A_end_idx].strip()
                            for pattern in ['option: (a)', 'option: a.','option: a ', 'option: a,', 'option a', '(a)', 'a:', 'cannot choose option b', ':a', ': a', A_answer]:
                                if sequences[i].lower().find(pattern.lower()) != -1:
                                    data[id+i]['Y'] = True
                                else: data[id+i]['Y'] = False
                        if self.task == 'QA' or self.task == 'RAG':
                            for i in range(len(sequences)):
                                print(data[id+i][self.input_key])
                                print(sequences[i])
                                print("======")
                                data[id+i][self.output_key] = sequences[i]
                else:
                    for i in range(len(sequences)):
                        print(data[id+i][self.input_key])
                        print(sequences[i])
                        print("======")
                        data[id+i][self.output_key] = sequences[i]

            output_list += data
            self.output_path = f"/workspace/wangjiatai/Swin-VIP/runs_qwen/Qwen_{self.mode}_is_context_{is_context}_{self.task}_no_SwinVIP.jsonl"
            #self.output_path = f"{self.output_path}/{self.model_name}_{'no_add'}.jsonl"
            with jsonlines.open(self.output_path, mode='w') as writer:
                writer.write_all(output_list)

    def plot_attention(self, data, data_name):
        if data_name == 'conflictQA-popQA-llama2-7b':
            y = [0 if item['Y'] else 1 for item in data]  
        elif data_name == 'TruthfulQA':
            y = [1 if item['single_evidence_true_false_label'] else 0 for item in data]  
        R = [[] for _ in range(len(self.model.model.layers))]
        plot_advanced_pie_chart(f"{self.output_pic_path}/{self.model_name}_{self.data_name}_label_distribution.png", y)
        for id in tqdm(range(0, len(data), self.batch_size)): #TODO len(data)
            encodings = self.tokenizer([i[self.input_key] for i in data[id:min(id+self.batch_size,len(data))]], return_tensors='pt', padding=True)
            y_batch = y[id:min(id+self.batch_size,len(y))]
            with torch.no_grad():
                _ = self.model(**encodings)
            tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
            if data_name == 'conflictQA-popQA-llama2-7b':
                start_id = [i.index('Information')+2 for i in tokenized]
                end_id = [i.index('Question')-2 for i in tokenized]
            elif data_name == 'TruthfulQA':
                start_id = []
                for i in range(len(tokenized)):
                        for j in range(len(tokenized[i])):
                            if tokenized[i][j] == 'Information':
                                if tokenized[i][j+1] == ':':
                                    start_id.append(j+1)
                                    break 
                end_id = [i.index('Options')-3 for i in tokenized]
            encodings = encodings.to(self.device)
            for i in range(len(self.model.model.layers)):
                attn_score = self.model.model.layers[i].self_attn.attn_score      #(24, 304, 4096)
                for idx in range(len(start_id)):
                    if data_name == 'conflictQA-popQA-llama2-7b':
                        head_score = attn_score[idx,start_id[idx]:end_id[idx]]
                    elif data_name == 'TruthfulQA':
                        head_score = attn_score[idx,start_id[idx]+1:end_id[idx]+1]
                    random_head_score = random_select(head_score)
                    random_head_score = np.mean(random_head_score, axis=0)
                    R[i].append((random_head_score, y_batch[idx]))
        layer_performance = []
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        for i in range(len(R)):
            X = np.array([score for score, _ in R[i]])
            y_layer = np.array([label for _, label in R[i]])
            clf = LogisticRegression(random_state=0)
            clf.fit(X, y_layer)
            y_pred = clf.predict(X)
            accuracy = accuracy_score(y_layer, y_pred)
            f1 = f1_score(y_layer, y_pred)
            layer_performance.append({
                'Layer': i + 1,
                'Accuracy': accuracy,
                'F1 Score': f1
            })
        # 输出每层的性能
        for performance in layer_performance:
            print(f"Layer {performance['Layer']}: Accuracy = {performance['Accuracy']:.4f}, F1 Score = {performance['F1 Score']:.4f}")

    def plot_mse(self, data, data_name):
        with open('/wangjiatai/weight/Swin-VIP/data/atten_score_dataset/conflict_llama2_val.pkl', 'rb') as f:
            R_loaded = pickle.load(f)
        if self.mode == 'mu':
            pass
        elif self.mode == 'montecarlo':
            montecarlo_samples = 10
        print('-------------------------------------')
        if data_name == 'conflictQA-popQA-llama2-7b':
            y = [0 if item['Y'] else 1 for item in data]  
        elif data_name == 'TruthfulQA':
            y = [1 if item['single_evidence_true_false_label'] else 0 for item in data]  
        models = []
        for i in range(len(self.model.model.layers)):  
            model = QDistributionVIB().to(self.device)
            weight_path = f'/wangjiatai/weight/Swin-VIP/checkpoint/TruthfulQA/200_VIB_layer_{i}.pth' 
            model.load_state_dict(torch.load(weight_path))
            models.append(model.eval())
        layer_mse_losses = [0.0] * len(self.model.model.layers)
        for id in tqdm(range(0, len(data), self.batch_size)): #TODO len(data)
            encodings = self.tokenizer([i[self.input_key] for i in data[id:min(id+self.batch_size,len(data))]], return_tensors='pt', padding=True)
            y_batch = y[id:min(id+self.batch_size,len(y))]
            with torch.no_grad():
                _ = self.model(**encodings)
            encodings = encodings.to(self.device)
            tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in encodings.input_ids]
            if data_name == 'conflictQA-popQA-llama2-7b':
                start_id = [i.index('Information')+2 for i in tokenized]
            elif data_name == 'TruthfulQA':
                start_id = []
                for i in range(len(tokenized)):
                        for j in range(len(tokenized[i])):
                            if tokenized[i][j] == 'Information':
                                if tokenized[i][j+1] == ':':
                                    start_id.append(j+1)
                                    break 
            for layer in range(len(self.model.model.layers)):
                random_head_scores_layer = R_loaded[layer]
                for idx in range(len(start_id)):
                    random_head_score = random_head_scores_layer[id + idx]
                    if self.mode=='mu':
                        predict_result = models[layer].detech_mu(torch.Tensor(random_head_score).to(self.device))
                    elif self.mode=='montecarlo':
                        predictions = []
                        for _ in range(montecarlo_samples):
                            y_pred, _, _ = models[layer](torch.Tensor(random_head_score).to(self.device))
                            predictions.append(y_pred)
                        predictions = torch.stack(predictions)
                        predict_result = predictions.mean(0)
                    mse_loss = F.mse_loss(predict_result.mean(), torch.tensor(y_batch[idx], dtype=torch.float).to(self.device))
                    layer_mse_losses[layer] += mse_loss.item()
                
        for i, loss in enumerate(layer_mse_losses):
            print(f"Layer {i + 1} MSE Loss: {loss}")




    

    

        
