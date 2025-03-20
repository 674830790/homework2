import jieba
import os
import string
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def preprocess(content, unit='word'):
    # 广告关键词过滤
    ad_keywords = ['www', 'cr173', 'com', '下载站', '电子书', '免费','txt','.']
    for kw in ad_keywords:
        content = content.replace(kw, '')
    
    # 定义标点符号（中英文均包括）
    punctuation = set(string.punctuation + '，。！？；：“”‘’（）【】…—\n\r\t ')
    
    if unit == 'word':
        tokens = jieba.lcut(content)
        filtered = [token for token in tokens if token not in punctuation]
    elif unit == 'char':
        # 保留汉字且过滤掉标点和空白字符
        filtered = [char for char in content 
                    if char not in punctuation 
                    and char.strip() != '' 
                    and '\u4e00' <= char <= '\u9fff']
    else:
        filtered = []
    return filtered

def generate_dataset(file_list, K, unit='word', total_samples=1000):
    dataset, labels = [], []
    n_novels = len(file_list)
    base_samples = total_samples // n_novels
    remainder = total_samples % n_novels

    for i, file_path in enumerate(file_list):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gb18030') as f:
                    content = f.read()
            except Exception as e:
                print(f"无法读取文件 {file_path}: {e}")
                continue

        tokens = preprocess(content, unit)
        if len(tokens) < K:
            print(f"文件 {file_path} 的 token 数不足 {K}，跳过该文件。")
            continue
        
        # 计算本小说应抽取的段落数
        samples_per_novel = base_samples + (1 if i < remainder else 0)
        N = len(tokens)
        # 计算步长以均匀采样
        step = max(1, (N - K) // (samples_per_novel - 1)) if samples_per_novel > 1 else 1
        
        paragraphs = []
        for j in range(0, N - K + 1, step):
            current_tokens = tokens[j:j+K]
            if not current_tokens:
                continue
            # 将 token 用空格连接，构成段落文本
            para = ' '.join(current_tokens)
            if para.strip() == '':
                continue
            paragraphs.append(para)
            if len(paragraphs) >= samples_per_novel:
                break
        # 如果段落数量不足，则随机复制已有段落补足
        while len(paragraphs) < samples_per_novel:
            if paragraphs:
                paragraphs.append(random.choice(paragraphs))
            else:
                break
        
        dataset.extend(paragraphs)
        label = os.path.basename(file_path).replace('.txt', '')
        labels.extend([label] * len(paragraphs))
    
    # 检查最终样本数是否达到总数要求
    if len(dataset) != total_samples:
        print(f"注意：生成的段落数量为 {len(dataset)}（目标 {total_samples}）")
    return dataset, labels

def evaluate_lda(X_topics, labels, n_splits=10, test_size=100):
    scores = []
    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    labels = np.array(labels)
    for train_idx, test_idx in rs.split(X_topics):
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(X_topics[train_idx], labels[train_idx])
        preds = clf.predict(X_topics[test_idx])
        scores.append(accuracy_score(labels[test_idx], preds))
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    # 设置存放小说的文件夹路径
    base_path = r"G:\F盘\大学课程资料\大四\自然语言处理\作业二"  # 修改为实际路径
    
    # 获取目录下所有txt文件
    novel_files = [
        os.path.join(base_path, fname) 
        for fname in os.listdir(base_path)
        if fname.endswith('.txt')
    ]
    
    # 实验参数
    K_values = [20, 100, 500, 1000, 3000 ]  # 段落 token 数
    T_values = [5, 10,20,50]                 # LDA 主题数
    units = ['word', 'char']               # 分词基本单元

    results = []
    
    for unit in units:
        for K in K_values:
            print(f"\n处理基本单元：{unit}，段落长度 K={K}")
            dataset, labels = generate_dataset(novel_files, K, unit, total_samples=1000)
            # 打印前 5 个样本作为案例
            novel_count = {}
            for label in labels:
             novel_count[label] = novel_count.get(label, 0) + 1
            print("每个小说生成的段落数量：", novel_count)

            print("样例段落及其标签：")
            for i in range(5):
                print(f"标签：{labels[i]}")
                print(f"内容：{dataset[i]}")
                print("="*40)

            if not dataset or len(dataset) < 1000:
                print(f"警告：生成的样本数量不足，当前样本数为 {len(dataset)}")
                continue

            if unit == 'char':
                vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
            else:
                vectorizer = CountVectorizer()
            
            X = vectorizer.fit_transform(dataset)
            print(f"CountVectorizer 特征矩阵大小：{X.shape}")
            
            for T in T_values:
                lda = LatentDirichletAllocation(n_components=T, random_state=42)
                X_topics = lda.fit_transform(X)
                mean_acc, std_acc = evaluate_lda(X_topics, labels, n_splits=10, test_size=100)
                results.append((unit, K, T, mean_acc, std_acc))
                print(f"【unit: {unit}, K: {K}, T: {T}】 分类准确率：{mean_acc:.4f} ± {std_acc:.4f}")
    
    # 最终结果存储在 results 列表中，可用于进一步分析和讨论
    print("\n实验结束，结果如下：")
    for res in results:
        unit, K, T, mean_acc, std_acc = res
        print(f"基本单元: {unit}, K: {K}, T: {T}, Acc: {mean_acc:.4f} ± {std_acc:.4f}")

        # 首先将 results 整理成一个结构化字典，便于按 unit、K、T 查询
acc_data = {}  # acc_data[unit][K][T] = mean_acc
for unit, K, T, mean_acc, std_acc in results:
    if unit not in acc_data:
        acc_data[unit] = {}
    if K not in acc_data[unit]:
        acc_data[unit][K] = {}
    acc_data[unit][K][T] = mean_acc

T_values = [5, 10, 20, 50]

# 依次为 unit='word' 和 unit='char' 各生成一张图
for current_unit in ['word', 'char']:
    # 创建画布
    plt.figure(figsize=(8,6))
    
    # 确保先按从小到大的顺序绘制不同的 K
    sorted_K_list = sorted(acc_data.get(current_unit, {}).keys())
    
    for K_val in sorted_K_list:
        mean_acc_list = []
        for T_val in T_values:
            if T_val in acc_data[current_unit][K_val]:
                mean_acc_list.append(acc_data[current_unit][K_val][T_val])
            else:
                mean_acc_list.append(0)
        
        # 绘制折线
        plt.plot(T_values, mean_acc_list, marker='o', label=f'K={K_val}')
    
    # 设置标题、坐标轴和图例
    plt.title(f"Accuracy vs. T (unit='{current_unit}')")
    plt.xlabel("Number of Topics (T)")
    plt.ylabel("Classification Accuracy")
    plt.ylim([0, 1.0])  
    plt.legend()
    plt.grid(True)  # 网格
    plt.tight_layout()
    
    plt.show()