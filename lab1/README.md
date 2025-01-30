## Lab1: Logistic Regression and Naive Bayes Text Classification Lab Report
#### **Objective**
The goal of this experiment is to perform a binary classification task on text data based on the `caffe.csv`, `incubator-mxnet.csv`, `keras.csv`, pytorch.csv, and `tensorflow.csv` datasets to predict whether the text is relevant to a specific category (e.g., bug reports, performance issues, etc.).

#### **Data preprocessing**

1. **Text cleanup**
    * Retain only ASCII characters, remove non-English characters (e.g. Chinese, Korean, etc.)
    * Remove all punctuation and special characters
2. **Text standardisation**
    * Segmentation using `word_tokenize`
    * `WordNetLemmatizer` for lemma reduction
    * Filter out the `stopwords` to remove the meaningless single
3. **Feature extraction**
    * Feature extraction with `TfidfVectorizer`
    * Filtering words longer than 2
4. **Custom word weights**
    * The high frequency terms were first filtered out using `min_df=20`, then weights were manually added to these high frequency terms and finally saved to `weights.csv`.

#### **Training**
* Multi-round experiment (30 times)
* 70 per cent training set, 30 per cent test set
* Training two different classification models for comparison
    * Logistic Regression
    * Naive Bayes

#### **Results**
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

Logistic Regression is the more suitable model for this task as it strikes a better balance between Precision and Recall.。


#### Thinking
* The documents have a small number of languages other than English, such as Chinese or Korean. This makes it challenging to face the classification problem, and it may be best to translate all languages into English during preprocessing, but there is a huge loss of speed and performance in calling the translation API.
