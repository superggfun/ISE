import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_ind
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk

# Download NLTK data (required for the first run)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    """Map NLTK POS tags to WordNet POS tags"""
    from nltk import pos_tag
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to noun


def preprocess_text(text):
    """Custom text preprocessing function"""
    # Remove non-English characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Keep only ASCII characters (0-127)

    # Tokenization
    tokens = word_tokenize(text)

    # Lemmatization
    processed_tokens = [
        lemmatizer.lemmatize(token.lower(), get_wordnet_pos(token.lower()))
        for token in tokens
        if token.isalpha() and token.lower() not in stop_words
    ]
    return " ".join(processed_tokens)


# Load and merge file data
file_path_prefix = "lab1/datasets/"  # Replace with the actual file path
file_names = ["caffe.csv", "incubator-mxnet.csv", "keras.csv", "pytorch.csv", "tensorflow.csv"]
file_paths = [file_path_prefix + file_name for file_name in file_names]

# Load data
dataframes = [pd.read_csv(file, encoding="utf-8") for file in file_paths]
df = pd.concat(dataframes, ignore_index=True)

# Data preprocessing
df["text"] = df["Title"].fillna("")  # Fill missing values
df["text"] = df["text"].apply(preprocess_text)  # Apply the custom text preprocessing function
df = df[df["class"].isin([0, 1])]  # Ensure the "class" column contains only 0 and 1

X = df["text"]  # Feature column
y = df["class"]  # Label column

# Define hyperparameters
NUM_EXPERIMENTS = 30  # Number of experiments
results_logistic = {"accuracy": [], "precision": [], "recall": [], "f1": []}
results_naive_bayes = {"accuracy": [], "precision": [], "recall": [], "f1": []}

# Load custom weights
custom_weights = {}
with open("lab1/weights.csv", "r", encoding="utf-8") as f:
    for line in f:
        word, weight = line.strip().split(",")
        custom_weights[word] = float(weight)

# Perform multiple experiments
for i in range(NUM_EXPERIMENTS):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i, stratify=y
    )

    # Feature extraction (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w{2,}\b")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Apply custom weights
    feature_names = vectorizer.get_feature_names_out()
    X_train_tfidf_weighted = X_train_tfidf.copy()
    X_test_tfidf_weighted = X_test_tfidf.copy()

    for word, weight in custom_weights.items():
        if word in feature_names:
            index = np.where(feature_names == word)[0][0]
            X_train_tfidf_weighted[:, index] *= weight
            X_test_tfidf_weighted[:, index] *= weight

    # Logistic Regression
    logistic_model = LogisticRegression(class_weight="balanced", solver="liblinear")
    logistic_model.fit(X_train_tfidf_weighted, y_train)
    y_pred_logistic = logistic_model.predict(X_test_tfidf_weighted)

    # Naive Bayes
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(X_train_tfidf_weighted, y_train)
    y_pred_naive_bayes = naive_bayes_model.predict(X_test_tfidf_weighted)

    # Record metrics
    results_logistic["accuracy"].append(accuracy_score(y_test, y_pred_logistic))
    results_logistic["precision"].append(precision_score(y_test, y_pred_logistic))
    results_logistic["recall"].append(recall_score(y_test, y_pred_logistic))
    results_logistic["f1"].append(f1_score(y_test, y_pred_logistic))

    results_naive_bayes["accuracy"].append(accuracy_score(y_test, y_pred_naive_bayes))
    results_naive_bayes["precision"].append(precision_score(y_test, y_pred_naive_bayes))
    results_naive_bayes["recall"].append(recall_score(y_test, y_pred_naive_bayes))
    results_naive_bayes["f1"].append(f1_score(y_test, y_pred_naive_bayes))

# Summarize results
print("\nLogistic Regression:")
for metric in results_logistic:
    mean_value = np.mean(results_logistic[metric])
    std_value = np.std(results_logistic[metric])
    print(f"  Mean {metric.capitalize()}: {mean_value:.4f}, Std: {std_value:.4f}")

print("\nNaive Bayes:")
for metric in results_naive_bayes:
    mean_value = np.mean(results_naive_bayes[metric])
    std_value = np.std(results_naive_bayes[metric])
    print(f"  Mean {metric.capitalize()}: {mean_value:.4f}, Std: {std_value:.4f}")
