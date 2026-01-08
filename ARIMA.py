import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# 1. 读取数据
# -----------------------------
file_path = "CN_Yield_2018_2025.xlsx"
df = pd.read_excel(file_path)

# 处理日期索引
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
else:
    df.index = pd.to_datetime(df.iloc[:, 0])
    df = df.iloc[:, 1:]

if '10Y' not in df.columns:
    raise ValueError("未找到 '10Y' 列，请检查列名")

y = df['10Y'].dropna().sort_index()


# -----------------------------
# 2. 按时间顺序划分训练集（80%）和测试集（20%）
# -----------------------------
n = len(y)
n_train = int(n * 0.8)
y_train = y.iloc[:n_train]
y_test = y.iloc[n_train:]

print(f"总数据点: {n}")
print(f"训练集: {len(y_train)} ({len(y_train)/n:.1%})")
print(f"测试集: {len(y_test)} ({len(y_test)/n:.1%})")


# -----------------------------
# 3. 自动选择 ARIMA(p,d,q) 阶数（仅基于训练集）
# -----------------------------
def find_best_arima_order(ts, max_p=3, max_q=3):
    best_aic = np.inf
    best_order = (0, 0, 0)
    
    # 检查平稳性以确定 d
    def is_stationary(x):
        p_val = adfuller(x.dropna())[1]
        return p_val <= 0.05
    
    d = 0 if is_stationary(ts) else 1

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = ARIMA(ts, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue
    return best_order

best_order = find_best_arima_order(y_train)
print(f"选定的 ARIMA 阶数（基于训练集）: {best_order}")


# -----------------------------
# 4. Walk-Forward 预测（逐点滚动预测）
# -----------------------------
predictions = []
history = list(y_train.values)

for i in range(len(y_test)):
    # 拟合模型
    model = ARIMA(history, order=best_order)
    model_fit = model.fit()
    # 预测下一步
    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)
    # 将真实值加入历史（模拟实际交易中已知新信息）
    history.append(y_test.iloc[i])



# -----------------------------
# 5. 构建结果 DataFrame
# -----------------------------
results = pd.DataFrame({
    'Date': y_test.index,
    'Actual_10Y_Yield': y_test.values,
    'Predicted_10Y_Yield': predictions
})
results.set_index('Date', inplace=True)


# -----------------------------
# 6. 保存结果
# -----------------------------
output_file = "ARIMA_forecast.csv"
results.to_csv(output_file)
print(f"\n✅ 样本外预测结果已保存至 '{output_file}'")


# 可选：计算 MAE、RMSE 等指标
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(results['Actual_10Y_Yield'], results['Predicted_10Y_Yield'])
rmse = np.sqrt(mean_squared_error(results['Actual_10Y_Yield'], results['Predicted_10Y_Yield']))
print(f"\n📊 测试集性能指标:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
