import numpy as np
import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# **1. 读取整个文件夹的 CSV 文件**
data_dir = 'lab2/datasets/batlik'  # 修改为你的实际路径
merged_data = pd.DataFrame()

for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)
        data = pd.read_csv(file_path)
        data['workload'] = file_name.replace('.csv', '')  # 使用文件名作为 workload
        merged_data = pd.concat([merged_data, data], ignore_index=True)

print(f"合并后的数据集大小: {merged_data.shape}")

# **2. 进行 One-Hot 编码**
encoder = OneHotEncoder(sparse_output=False)
workload_encoded = encoder.fit_transform(merged_data[['workload']])
workload_encoded_df = pd.DataFrame(workload_encoded, columns=encoder.get_feature_names_out(['workload']))
merged_data = pd.concat([merged_data, workload_encoded_df], axis=1)
merged_data.drop(columns=['workload'], inplace=True)  # 删除原 workload 列

# **3. 使用 K-Means 聚类进行动态分组**
kmeans = KMeans(n_clusters=2, random_state=42)  # 选择 2 组
merged_data['Group'] = kmeans.fit_predict(merged_data[['time']])

# 显示分组情况
group_counts = merged_data['Group'].value_counts()
print("K-Means 动态分组结果：")
print(group_counts)

# **4. 分离特征和目标值**
X = merged_data.drop(columns=['time', 'Group'])  # 输入特征
y = merged_data['time']  # 目标值

# **按 K-Means 预测的组划分训练集和测试集**
X_group1 = X[merged_data['Group'] == 0]
y_group1 = y[merged_data['Group'] == 0]
X_group2 = X[merged_data['Group'] == 1]
y_group2 = y[merged_data['Group'] == 1]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_group1, y_group1, test_size=0.3, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_group2, y_group2, test_size=0.3, random_state=42)

print(f"Group 1 训练集大小: {X_train1.shape}, 测试集大小: {X_test1.shape}")
print(f"Group 2 训练集大小: {X_train2.shape}, 测试集大小: {X_test2.shape}")

# **5. 训练数据清理**
def clean_training_data(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    
    # 计算误差并删除偏离较大的点
    errors = np.abs(y_train - y_train_pred)
    threshold = errors.mean() + 2 * errors.std()
    clean_indices = errors <= threshold
    return X_train[clean_indices], y_train[clean_indices]

# 清洗训练数据
X_train1_cleaned, y_train1_cleaned = clean_training_data(X_train1, y_train1)
X_train2_cleaned, y_train2_cleaned = clean_training_data(X_train2, y_train2)

print(f"清洗后 Group 1 训练集大小: {X_train1_cleaned.shape}")
print(f"清洗后 Group 2 训练集大小: {X_train2_cleaned.shape}")

# **6. 训练组合模型**
class CombinedModel:
    def __init__(self):
        self.model1 = LinearRegression()
        self.model2 = LinearRegression()
    
    def fit(self, X_group1, y_group1, X_group2, y_group2):
        self.model1.fit(X_group1, y_group1)
        self.model2.fit(X_group2, y_group2)
    
    def predict(self, X, data_group):
        predictions = []
        for idx, row in X.iterrows():
            row_df = row.to_frame().T  # 转换为 DataFrame 以兼容 sklearn
            if data_group.loc[idx] == 0:
                predictions.append(self.model1.predict(row_df)[0])
            else:
                predictions.append(self.model2.predict(row_df)[0])
        return np.array(predictions)

# 训练组合模型
combined_model = CombinedModel()
combined_model.fit(X_train1_cleaned, y_train1_cleaned, X_train2_cleaned, y_train2_cleaned)

# **7. 预测测试数据**
X_test_combined = pd.concat([X_test1, X_test2])
y_test_combined = pd.concat([y_test1, y_test2])
group_test_combined = merged_data.loc[X_test_combined.index, 'Group']

y_pred_combined = combined_model.predict(X_test_combined, group_test_combined)

# **8. 评估模型性能**
mae_combined = mean_absolute_error(y_test_combined, y_pred_combined)
mape_combined = mean_absolute_percentage_error(y_test_combined, y_pred_combined)
rmse_combined = np.sqrt(mean_squared_error(y_test_combined, y_pred_combined))

print("\nK-Means 组合模型评估：")
print(f"MAE: {mae_combined}, MAPE: {mape_combined}, RMSE: {rmse_combined}")

