import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import os

# 创建保存图片的文件夹
output_folder = '/Users/ellie/Desktop/png'
os.makedirs(output_folder, exist_ok=True)

# 加载数据
data = pd.read_csv('/Users/ellie/Documents/Assets/csv/train.csv')

print("\n数据信息:")
print(data.info())
print("\n缺失值统计:")
print(data.isnull().sum())

# 数据预处理
features = ['age', 'job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_index', 'cons_conf_index', 'lending_rate3m', 'nr_employed']
target = 'subscribe'

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# 数值变量
numeric_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 
                'cons_price_index', 'cons_conf_index', 'lending_rate3m', 'nr_employed']

# 对二元分类变量进行编码
binary_cols = ['default', 'housing', 'loan', 'subscribe']
for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# 对其他分类变量使用标签编码
label_encoders = {}
for col in ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 分离特征和目标变量
X = data[features]
y = data[target]

# 5. 处理数值变量
# 标准化数值特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[numeric_cols])
data[numeric_cols] = scaled_features

# 定义数值特征和分类特征
numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_index', 'cons_conf_index', 'lending_rate3m', 'nr_employed']
categorical_features = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# 定义预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 应用预处理步骤
X_preprocessed = preprocessor.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# 使用 SMOTE 处理数据不平衡
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 定义 XGBoost 模型
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 5,  # 为少数类分配更高的权重
    'random_state': 42
}

# 定义模型管道
pipeline = Pipeline(steps=[
    ('classifier', XGBClassifier(**xgb_params))
])

# 训练模型
pipeline.fit(X_train_resampled, y_train_resampled)

# 预测测试集
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# 评估模型
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# 可视化部分
# 1. 目标变量分布
plt.figure(figsize=(8, 6))
sns.countplot(x=target, data=data)
plt.title('Target Variable Distribution')
plt.savefig(os.path.join(output_folder, "target_distribution.png"))
plt.close()

# 2. 特征重要性
feature_importances = pipeline.named_steps['classifier'].feature_importances_
feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out())
sorted_idx = feature_importances.argsort()[::-1]

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances[sorted_idx], y=[feature_names[i] for i in sorted_idx])
plt.title('Feature Importances')
plt.savefig(os.path.join(output_folder, "feature_importances.png"))
plt.close()

# 3. 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
plt.close()

# 4. ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_folder, "roc_curve.png"))
plt.close()