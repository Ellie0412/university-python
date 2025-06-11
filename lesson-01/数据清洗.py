import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# 加载数据
data = pd.read_csv('/Users/ellie/Desktop/第10次课 特征学习/bank+marketing/bank-full.csv', sep=';')
data.to_csv('bank-full-cleaned.csv', index=False)

# 查看数据的基本信息
print(data.info())
print(data.describe())

# # 定性数据可视化
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
for col in categorical_columns:
    plt.figure(figsize=(15, 5))
    sns.countplot(x=col, data=data)
    plt.title(f'Distribution of {col}')
    plt.show()

# # 定量数据可视化
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for col in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# 描述性统计
print(data.describe(include='all'))

# 检查缺失值
print(data.isnull().sum())


# 计算四分位数
Q1 = data['balance'].quantile(0.25)  # 第25百分位数
Q3 = data['balance'].quantile(0.75)  # 第75百分位数

# 计算四分位距
IQR = Q3 - Q1

# 确定异常值的范围
lower_bound = Q1 - 1.5 * IQR  # 下界
upper_bound = Q3 + 1.5 * IQR  # 上界

# 筛选数据：保留那些在下界和上界之间的数据
data = data[(data['balance'] >= lower_bound) & (data['balance'] <= upper_bound)]

# 检查处理后的数据
print("处理后的数据:")
print(data.describe())

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures,StandardScaler

# 定性数据独热编码
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_data = encoder.fit_transform(data[categorical_columns])
categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

# 定量数据多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
numerical_data = poly.fit_transform(data[numerical_columns])
numerical_df = pd.DataFrame(numerical_data, columns=poly.get_feature_names_out(numerical_columns))

# 合并数据
data_encoded = pd.concat([numerical_df, categorical_df], axis=1)
# 标准化数据
scaler = StandardScaler()
data_encoded_scaled = scaler.fit_transform(data_encoded)

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 目标变量
y = data['y'].map({'no': 0, 'yes': 1})  # 将目标变量转换为数值

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data_encoded, y, test_size=0.2, random_state=42)

# 训练随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 获取特征重要性
feature_importances = rf.feature_importances_

# 获取特征名称
# 注意：这里的特征名称需要区分原始特征和经过处理后的特征
# 原始特征名称
original_feature_names = numerical_columns + categorical_columns

# 经过处理后的特征名称
# 多项式特征扩展后的名称
poly_feature_names = poly.get_feature_names_out(numerical_columns)

# 独热编码后的名称
onehot_feature_names = encoder.get_feature_names_out(categorical_columns)

# 合并后的特征名称
all_feature_names = list(poly_feature_names) + list(onehot_feature_names)

# 创建特征重要性 DataFrame
feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 打印重要特征
print("\nFeature Importances:")
print(feature_importance_df.head(10))

# 绘制特征重要性条形图
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances')
plt.show()



from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)

# LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

print("LDA Explained Variance Ratio:", lda.explained_variance_ratio_)

# 训练随机森林分类器
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)

# 评估模型性能
y_pred_pca = rf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("Accuracy with PCA:", accuracy_pca)

# 训练随机森林分类器
rf_lda = RandomForestClassifier(n_estimators=100, random_state=42)
rf_lda.fit(X_train_lda, y_train)

# 评估模型性能
y_pred_lda = rf_lda.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print("Accuracy with LDA:", accuracy_lda)



# 1. 不做特征选择和特征转换
# 使用原始数据进行训练和测试
logreg_original = LogisticRegression(max_iter=1000, random_state=42)
logreg_original.fit(X_train, y_train)
y_pred_original = logreg_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
print("Accuracy without feature selection or transformation:", accuracy_original)

# 2. 仅做特征选择
# 使用 SelectKBest 选择重要特征
selector = SelectKBest(score_func=f_classif, k=20)  # 选择前 20 个重要特征
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

logreg_selected = LogisticRegression(max_iter=1000, random_state=42)
logreg_selected.fit(X_train_selected, y_train)
y_pred_selected = logreg_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with feature selection only:", accuracy_selected)

# 3. 做特征选择和特征转换
# 使用 PCA 进行特征转换
pca = PCA(n_components=20)  # 选择 20 个主成分
X_train_pca = pca.fit_transform(X_train_selected)
X_test_pca = pca.transform(X_test_selected)

logreg_pca = LogisticRegression(max_iter=1000, random_state=42)
logreg_pca.fit(X_train_pca, y_train)
y_pred_pca = logreg_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("Accuracy with feature selection and PCA:", accuracy_pca)

# 使用 LDA 进行特征转换
lda = LinearDiscriminantAnalysis(n_components=1)  # 选择 1 个主成分
X_train_lda = lda.fit_transform(X_train_selected, y_train)
X_test_lda = lda.transform(X_test_selected)

logreg_lda = LogisticRegression(max_iter=1000, random_state=42)
logreg_lda.fit(X_train_lda, y_train)
y_pred_lda = logreg_lda.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print("Accuracy with feature selection and LDA:", accuracy_lda)