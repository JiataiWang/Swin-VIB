import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_advanced_pie_chart(save_path, y):
    # 创建文件夹（如果不存在）
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    # 计算0和1的数量
    data = pd.Series(y).value_counts().reset_index()
    data.columns = ['label', 'count']
    data['label'] = data['label'].map({0: 'Error', 1: 'Correct'})
    
    # 设置主题
    sns.set_theme(style="whitegrid")
    
    # 绘制饼图
    plt.figure(figsize=(8, 8))
    colors = sns.color_palette("pastel")[0:2]
    plt.pie(data['count'], labels=data['label'], autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('No conflict in LLM to answer the right or wrong', fontsize=16)
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
