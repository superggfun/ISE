## Lab1: Logistic Regression 与 Naive Bayes 文本分类实验报告
#### **任务目标**
本实验的目标是基于 `caffe.csv`、`incubator-mxnet.csv`、`keras.csv`、pytorch.csv 和`tensorflow.csv` 数据集，对文本数据进行二分类任务，来预测文本是否与特定类别（如错误报告、性能问题等）相关。

#### **数据预处理**

1. **文本清理**
    * 仅保留 ASCII 字符，去除非英文字符（如中文、韩文等）
    * 移除所有标点符号和特殊字符
2. **文本标准化**
    * 使用 `word_tokenize` 进行分词
    * 采用 `WordNetLemmatizer` 进行 词形还原
    * 过滤掉 NLTK stopwords，去除无意义单
3. **特征提取**
    * 采用 `TfidfVectorizer` 进行特征提取
    * 过滤长度大于2的单词
4. **自定义词权重**
    * 首先使用`min_df=20`过滤出高频词汇，然后对这些高频词汇进行手动增加权重，并最后保存到`weights.csv`

#### **训练**
* 多轮实验（30 次随机划分数据）
* 70% 训练集，30% 测试集
* 训练两个不同分类模型进行比较
    * Logistic Regression
    * Naive Bayes

#### **实验结果**
1. **Logistic Regression**:
    * **Mean Accuracy**: 0.9087 ± 0.0058
    * **Mean Precision**: 0.6873 ± 0.0211
    * **Mean Recall**: 0.8115 ± 0.0210
    * **Mean F1-Score**: 0.7439 ± 0.0135

2. **Naive Bayes**:
    * **Mean Accuracy:** 0.9061 ± 0.0049
    * **Mean Precision:** 0.8671 ± 0.0334
    * **Mean Recall:** 0.5033 ± 0.0273
    * **Mean F1-Score:** 0.6362 ± 0.0225

Logistic Regression 是更适合该任务的模型，因为它在 Precision 和 Recall 之间取得了更好的平衡。


#### **思考**
* 文档中有少量除英语之外的语言，例如中文或者韩文。在面对分类问题时具有挑战，可能最好的办法是在预处理的时候把所有语言都翻译成英文，但是在调用翻译API又有巨大的速度性能上的损失。