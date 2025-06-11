import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import seaborn as sns
import os
# 忽略警告信息
warnings.filterwarnings("ignore")

# 读取数据
file_path = '/Users/ellie/Desktop/China_Tech_HK.csv'  # 假设文件路径正确
data = pd.read_csv(file_path)
# 获取所有股票代码
stocks = data['ts_code'].unique()
# 1. 数据检查
# 检查缺失值
print(data.isnull().sum())

# 检查数据类型
print(data.dtypes)


# 处理异常值：删除价格为负数的行
data = data[data['open'] >= 0]
data = data[data['high'] >= 0]
data = data[data['low'] >= 0]
data = data[data['close'] >= 0]

# 3. 数据转换
# 日期格式转换
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data['year_month'] = data['trade_date'].dt.to_period('M')

# 数值格式转换
data['open'] = pd.to_numeric(data['open'], errors='coerce')
data['high'] = pd.to_numeric(data['high'], errors='coerce')
data['low'] = pd.to_numeric(data['low'], errors='coerce')
data['close'] = pd.to_numeric(data['close'], errors='coerce')
data['vol'] = pd.to_numeric(data['vol'], errors='coerce')
data['amount'] = pd.to_numeric(data['amount'], errors='coerce')

# 定义需要绘制的列
columns_to_plot = ['open', 'high', 'low', 'close', 'vol', 'amount']

# 为每只股票绘制分布图像
for stock in stocks:
    # 获取该股票的数据
    stock_data = data[data['ts_code'] == stock]
    
    # 如果该股票数据量太少，跳过
    if len(stock_data) < 10:
        print(f"股票 {stock} 的数据量不足，跳过")
        continue
    
    # 绘制直方图
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns_to_plot, 1):
        plt.subplot(3, 2, i)
        sns.histplot(stock_data[column].dropna(), kde=True, bins=30)
        plt.title(f'{stock} - {column} 分布')
        plt.xlabel(column)
        plt.ylabel('频率')
    plt.suptitle(f'股票 {stock} 的变量分布直方图', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # # 保存直方图
    # histogram_path = os.path.join('/Users/ellie/Desktop', f'{stock}_histogram.png')
    # plt.savefig(histogram_path)
    # plt.close()
# 4. 数据标准化或归一化
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化
scaler = StandardScaler()
data[['vol', 'amount']] = scaler.fit_transform(data[['vol', 'amount']])

# 归一化
scaler = MinMaxScaler()
data[['open', 'high', 'low', 'close']] = scaler.fit_transform(data[['open', 'high', 'low', 'close']])

# 计算月度VWAP(成交量加权平均价)
def calculate_vwap(group):
    return np.sum(group['close'] * group['vol']) / np.sum(group['vol'])

monthly_vwap = data.groupby(['ts_code', 'year_month']).apply(calculate_vwap).unstack(level=0)

# 获取所有股票代码
stocks = monthly_vwap.columns

# 为每只股票创建预测图表
for stock in stocks:
    try:
        # 获取该股票的VWAP数据
        stock_data = monthly_vwap[stock].dropna()
        
        # 检查数据量是否足够
        if len(stock_data) < 10:
            print(f"股票 {stock} 的数据量不足，跳过")
            continue
        
        # 准备训练数据
        history = stock_data.values
        
        # 训练ARIMA模型 (参数根据AIC自动选择)
        best_aic = np.inf
        best_order = None
        best_model = None
        
        # 参数网格搜索
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(history, order=(p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                            best_model = model_fit
                    except:
                        continue
        
        if best_model is None:
            print(f"无法为股票 {stock} 找到合适模型")
            continue
            
        # 预测未来3个月
        forecast = best_model.forecast(steps=3)
        
        # 创建时间索引
        last_date = stock_data.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=3, freq='M')
        
        # 绘制图表
        plt.figure(figsize=(10, 6))
        
        # 绘制历史数据
        plt.plot(stock_data.index.astype(str), history, 
                 label='历史数据(VWAP)', color='blue', marker='o')
        
        # 绘制预测数据
        plt.plot(future_dates.astype(str), forecast, 
                 label='预测数据', color='red', linestyle='--', marker='x')
        
        plt.title(f'股票 {stock} 价格预测 (基于月度VWAP)')
        plt.xlabel('年月')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        
        # 使用交叉验证计算模型评估指标
        cv_scores = []
        n_folds = 5
        split_size = len(history) // n_folds
        
        for fold in range(n_folds):
            train = history[:fold*split_size].tolist() + history[(fold+1)*split_size:].tolist()
            test = history[fold*split_size:(fold+1)*split_size].tolist()
            
            if len(train) < 2 or len(test) < 1:
                continue
            
            model = ARIMA(train, order=best_order)
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            rmse = sqrt(mean_squared_error(test, predictions))
            cv_scores.append(rmse)
        
        if cv_scores:
            cv_rmse = np.mean(cv_scores)
        else:
            cv_rmse = np.nan
        
        # 打印交叉验证结果
        print(f"股票 {stock} 的交叉验证 RMSE: {cv_rmse:.2f}")
        
        # 显示模型评估指标
        plt.text(0.02, 0.95, f'CV RMSE: {cv_rmse:.2f}\nARIMA{best_order}', 
                 transform=plt.gca().transAxes, fontsize=9)
        
        # # 保存或显示图表
        # plt.savefig(f'{stock}_forecast.png')  # 保存为图片
        # plt.show()  # 显示图表
        
    except Exception as e:
        print(f"处理股票 {stock} 时出错: {str(e)}")
        continue