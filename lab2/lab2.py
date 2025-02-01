import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# 1. 读取数据
data = pd.read_csv('lab2/datasets/batlik/corona.csv')
if 'time' not in data.columns:
    raise ValueError("数据集中必须包含 'time' 列")

# 2. **使用 K-Means 进行自动分组**
kmeans = KMeans(n_clusters=2, random_state=42)
X_features = data.drop(columns=['time'])  # 只使用输入特征进行聚类
data['Group'] = kmeans.fit_predict(X_features)

# 查看每个分组的样本数
group_counts = data['Group'].value_counts()
print("K-Means 动态分组结果：")
print(group_counts)

# 3. **分离输入特征和目标值**
X = data.drop(columns=['time', 'Group'])  # 只保留输入特征
y = data['time']  # 目标值

# 按 K-Means 分组划分训练集和测试集
X_group1 = X[data['Group'] == 0]
y_group1 = y[data['Group'] == 0]
X_group2 = X[data['Group'] == 1]
y_group2 = y[data['Group'] == 1]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_group1, y_group1, test_size=0.3, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_group2, y_group2, test_size=0.3, random_state=42)

print(f"Group 1 训练集大小: {X_train1.shape}, 测试集大小: {X_test1.shape}")
print(f"Group 2 训练集大小: {X_train2.shape}, 测试集大小: {X_test2.shape}")

# 4. **定义清洗训练集的函数**
def clean_training_data(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    
    # 计算误差
    errors = np.abs(y_train - y_train_pred)
    
    # 选择一种误差清理方法
    #threshold = np.percentile(errors, 70)  # 删除误差前 30% 的点
    threshold = errors.mean() + 2 * errors.std()  # 另一种方法
    
    clean_indices = errors <= threshold
    return X_train[clean_indices], y_train[clean_indices]

# **清洗 Group 1 和 Group 2 的训练集**
X_train1_cleaned, y_train1_cleaned = clean_training_data(X_train1, y_train1)
X_train2_cleaned, y_train2_cleaned = clean_training_data(X_train2, y_train2)

print(f"清洗后 Group 1 训练集大小: {X_train1_cleaned.shape}")
print(f"清洗后 Group 2 训练集大小: {X_train2_cleaned.shape}")

# 5. **定义组合模型**
class CombinedModel:
    def __init__(self, kmeans):
        self.kmeans = kmeans
        self.model1 = LinearRegression()
        self.model2 = LinearRegression()
    
    def fit(self, X_group1, y_group1, X_group2, y_group2):
        self.model1.fit(X_group1, y_group1)
        self.model2.fit(X_group2, y_group2)
    
    def predict(self, X):
        predictions = []
        groups = self.kmeans.predict(X)  # **使用 X 进行 K-Means 预测**
        
        for i, row in enumerate(X.iterrows()):  # **修正索引问题**
            row_df = row[1].to_frame().T  # 获取 DataFrame 格式
            if groups[i] == 0:
                predictions.append(self.model1.predict(row_df)[0])
            else:
                predictions.append(self.model2.predict(row_df)[0])
        return np.array(predictions)

# 训练组合模型
combined_model = CombinedModel(kmeans)
combined_model.fit(X_train1_cleaned, y_train1_cleaned, X_train2_cleaned, y_train2_cleaned)

# **获取测试数据**
X_test_combined = pd.concat([X_test1, X_test2])
y_test_combined = pd.concat([y_test1, y_test2])

# **使用组合模型进行预测**
y_pred_combined = combined_model.predict(X_test_combined)

# 6. **评估模型性能**
mae_combined = mean_absolute_error(y_test_combined, y_pred_combined)
mape_combined = mean_absolute_percentage_error(y_test_combined, y_pred_combined)
rmse_combined = np.sqrt(mean_squared_error(y_test_combined, y_pred_combined))

print("\nK-Means 组合模型评估：")
print(f"MAE: {mae_combined}, MAPE: {mape_combined}, RMSE: {rmse_combined}")
