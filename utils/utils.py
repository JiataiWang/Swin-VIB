import jsonlines
def read_data(path):
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return data
import numpy as np
import os

def random_select(matrix):
    n, _ = (matrix).shape
    if n <= 7:
        # 如果 n 小于等于 7，取全部内容
        return matrix
    else:
        # 随机选择一个开始位置，确保可以取到连续的 7 个条目
        start_index = np.random.randint(0, n - 7 + 1)
        return matrix[start_index:start_index + 7]
    
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")