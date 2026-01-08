import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# 1. 读取数据
def load_data(file_path):
    """
    读取国债收益率数据
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 打印列名以便调试
        print("数据列名:", df.columns.tolist())
        
        # 尝试自动识别日期列
        date_columns = []
        for col in df.columns:
            col_lower = str(col).lower()
            if 'date' in col_lower or '时间' in col_lower or '日期' in col_lower:
                date_columns.append(col)
        
        if date_columns:
            date_col = date_columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            print(f"使用列 '{date_col}' 作为日期索引")
        else:
            # 尝试第一列作为日期
            date_col = df.columns[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                print(f"使用第一列 '{date_col}' 作为日期索引")
            except:
                # 如果没有日期列，创建整数索引
                df.index = pd.date_range(start='2018-01-01', periods=len(df), freq='D')
                print("创建默认日期索引")
        
        # 查找10年期数据列
        yield_columns = []
        for col in df.columns:
            col_str = str(col)
            if '10' in col_str or '十年' in col_str or '10年' in col_str:
                yield_columns.append(col)
        
        if yield_columns:
            # 优先选择包含10Y的列
            for col in yield_columns:
                if '10Y' in str(col).upper() or '10Y' in str(col):
                    df = df[[col]].copy()
                    df.columns = ['10Y']
                    print(f"使用列 '{col}' 作为10年期国债收益率数据")
                    break
            else:
                df = df[[yield_columns[0]]].copy()
                df.columns = ['10Y']
                print(f"使用列 '{yield_columns[0]}' 作为10年期国债收益率数据")
        else:
            # 如果找不到10年期数据，使用第一列数值数据
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df = df[[numeric_cols[0]]].copy()
                df.columns = ['10Y']
                print(f"使用数值列 '{numeric_cols[0]}' 作为10年期国债收益率数据")
            else:
                raise ValueError("未找到合适的数值列作为收益率数据")
        
        # 确保数据按日期排序
        df.sort_index(inplace=True)
        
        # 检查缺失值
        missing_count = df['10Y'].isnull().sum()
        if missing_count > 0:
            print(f"警告: 发现 {missing_count} 个缺失值，将使用前向填充")
            df['10Y'].fillna(method='ffill', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        raise

# 2. 创建特征和标签
def create_features(df, lookback_days=20):
    """
    创建特征集：使用过去lookback_days天的收益率作为特征
    预测下一天的收益率
    """
    df_clean = df.copy()
    
    # 原始收益率
    df_clean['yield'] = df_clean['10Y']
    
    # 创建滞后特征
    for i in range(1, lookback_days + 1):
        df_clean[f'yield_lag_{i}'] = df_clean['yield'].shift(i)
    
    # 收益率变化率
    df_clean['return'] = df_clean['yield'].pct_change()
    for i in range(1, lookback_days + 1):
        df_clean[f'return_lag_{i}'] = df_clean['return'].shift(i)
    
    # 技术指标
    for window in [5, 10, 20]:
        df_clean[f'MA_{window}'] = df_clean['yield'].rolling(window=window).mean()
        df_clean[f'return_MA_{window}'] = df_clean['return'].rolling(window=window).mean()
        df_clean[f'volatility_{window}'] = df_clean['return'].rolling(window=window).std()
    
    # 动量指标
    df_clean['momentum_5'] = df_clean['yield'] - df_clean['yield'].shift(5)
    df_clean['momentum_10'] = df_clean['yield'] - df_clean['yield'].shift(10)
    
    # 目标变量：下一天的收益率
    df_clean['target_yield'] = df_clean['yield'].shift(-1)
    
    # 目标变量：下一天的收益率变化方向（用于分类问题，如果需要）
    df_clean['target_direction'] = (df_clean['target_yield'] > df_clean['yield']).astype(int)
    
    # 删除包含NaN的行
    df_clean.dropna(inplace=True)
    
    # 特征和目标分离
    feature_cols = [col for col in df_clean.columns 
                   if col not in ['10Y', 'yield', 'target_yield', 'target_direction', 'return']]
    
    X = df_clean[feature_cols]
    y_reg = df_clean['target_yield']  # 回归目标
    y_cls = df_clean['target_direction']  # 分类目标
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"特征示例: {feature_cols[:5]}...")
    print(f"样本数量: {len(X)}")
    
    return X, y_reg, y_cls, df_clean

# 3. 构建和训练模型
def build_and_train_models(X, y_reg):
    """
    构建并训练多个回归模型
    """
    # 划分训练集和测试集（按时间顺序）
    split_idx = int(len(X) * 0.8)  # 80%训练，20%测试
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集时间范围: {X_train.index.min()} 到 {X_train.index.max()}")
    print(f"测试集时间范围: {X_test.index.min()} 到 {X_test.index.max()}")
    
    # 模型定义 - 全部使用回归模型
    models = {
        'LinearRegression': LinearRegression(),
        'BayesianRidge': BayesianRidge(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(100, 50), 
            max_iter=1000, 
            random_state=42,
            early_stopping=True,
            learning_rate_init=0.001
        ),
        'RF': RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
        ),
        'SVM': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    # 存储结果
    model_results = {}
    
    # 训练每个模型
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        
        try:
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 存储结果
            model_results[name] = {
                'model': model,
                'predictions': y_pred,
                'actual': y_test.values,
                'dates': y_test.index,
                'X_test': X_test,
                'y_test': y_test
            }
            
            # 计算评价指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 计算方向准确性
            if len(y_test) > 1:
                # 实际方向
                actual_direction = np.sign(np.diff(y_test))
                # 预测方向
                pred_direction = np.sign(np.diff(y_pred))
                direction_accuracy = np.mean(actual_direction == pred_direction)
            else:
                direction_accuracy = np.nan
            
            print(f"{name} 模型结果:")
            print(f"  均方误差 (MSE): {mse:.6f}")
            print(f"  均方根误差 (RMSE): {rmse:.6f}")
            print(f"  平均绝对误差 (MAE): {mae:.6f}")
            print(f"  R²分数: {r2:.4f}")
            if not np.isnan(direction_accuracy):
                print(f"  方向准确性: {direction_accuracy:.2%}")
            
        except Exception as e:
            print(f"训练 {name} 模型时出错: {str(e)}")
            continue
    
    return model_results, X_test, y_test

# 4. 保存预测结果
def save_predictions(model_results, output_file='yield_predictions.csv'):
    """
    保存所有模型的预测结果
    """
    if not model_results:
        print("没有模型结果可保存")
        return None
    
    # 使用第一个模型的日期作为基准
    first_model = list(model_results.keys())[0]
    dates = model_results[first_model]['dates']
    
    # 创建DataFrame保存结果
    results_df = pd.DataFrame({'Date': dates})
    
    for name, result in model_results.items():
        if len(result['predictions']) != len(dates):
            print(f"警告: {name}模型的预测长度不匹配")
            continue
            
        results_df[f'{name}_Actual'] = result['actual']
        results_df[f'{name}_Predicted'] = result['predictions']
        results_df[f'{name}_Error'] = result['actual'] - result['predictions']
    
    # 保存到CSV
    results_df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")
    
    # 显示前几行
    print("\n预测结果预览:")
    print(results_df.head())
    
    return results_df

# 5. 模型评价报告
def generate_evaluation_report(model_results):
    """
    生成模型评价报告
    """
    if not model_results:
        print("没有模型结果可生成报告")
        return None
    
    report = []
    
    for name, result in model_results.items():
        y_true = result['actual']
        y_pred = result['predictions']
        
        # 计算各种指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算方向准确性
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            direction_accuracy = np.mean(actual_direction == pred_direction)
        else:
            direction_accuracy = np.nan
        
        # 计算平均绝对百分比误差
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        report.append({
            'Model': name,
            'MSE': f"{mse:.6f}",
            'RMSE': f"{rmse:.6f}",
            'MAE': f"{mae:.6f}",
            'MAPE': f"{mape:.2f}%",
            'R²': f"{r2:.4f}",
            'Direction_Accuracy': f"{direction_accuracy:.2%}" if not np.isnan(direction_accuracy) else 'N/A'
        })
    
    # 转换为DataFrame并按R²排序
    report_df = pd.DataFrame(report)
    report_df['R²_Value'] = report_df['R²'].str.replace('%', '').astype(float)
    report_df = report_df.sort_values('R²_Value', ascending=False).drop('R²_Value', axis=1)
    
    # 保存报告
    report_df.to_csv('model_evaluation_report.csv', index=False)
    print("模型评价报告已保存到: model_evaluation_report.csv")
    
    # 打印报告
    print("\n" + "="*80)
    print("模型性能评价报告")
    print("="*80)
    print(report_df.to_string(index=False))
    
    return report_df

# 6. 为交易信号做准备的数据处理
def prepare_for_trading_signals(model_results, results_df):
    """
    为制作交易信号准备数据
    返回包含预测收益率、预测涨跌概率等信息的DataFrame
    """
    if not model_results:
        print("没有模型结果，无法准备交易信号")
        return None
    
    # 选择R²最高的模型作为主要模型
    best_model_name = None
    best_r2 = -np.inf
    
    for name, result in model_results.items():
        y_true = result['actual']
        y_pred = result['predictions']
        r2 = r2_score(y_true, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
    
    print(f"\n选择最佳模型进行交易信号生成: {best_model_name} (R² = {best_r2:.4f})")
    
    # 获取最佳模型的预测结果
    best_result = model_results[best_model_name]
    y_pred = best_result['predictions']
    y_true = best_result['actual']
    dates = best_result['dates']
    
    # 创建交易信号DataFrame
    trading_df = pd.DataFrame({
        'Date': dates,
        'Actual_Yield': y_true,
        'Predicted_Yield': y_pred
    })
    
    # 计算预测的收益率变化
    trading_df['Predicted_Change'] = trading_df['Predicted_Yield'].diff()
    
    # 为了避免第一个值为NaN，使用0填充
    trading_df['Predicted_Change'].fillna(0, inplace=True)
    
    # 预测涨跌信号（1表示看涨，-1表示看跌，0表示不变）
    trading_df['Signal'] = np.sign(trading_df['Predicted_Change'])
    
    # 预测上涨概率（基于预测变化的大小）
    # 使用sigmoid函数将变化值映射到[0,1]区间
    change_std = trading_df['Predicted_Change'].std()
    if change_std > 0:
        # 标准化变化值
        normalized_change = trading_df['Predicted_Change'] / (change_std * 2)
        # 使用sigmoid函数
        trading_df['Up_Probability'] = 1 / (1 + np.exp(-normalized_change))
    else:
        trading_df['Up_Probability'] = 0.5
    
    # 添加实际涨跌方向（用于评估）
    trading_df['Actual_Change'] = trading_df['Actual_Yield'].diff()
    trading_df['Actual_Change'].fillna(0, inplace=True)
    trading_df['Actual_Direction'] = np.sign(trading_df['Actual_Change'])
    
    # 计算信号准确性
    trading_df['Signal_Correct'] = (trading_df['Signal'] == trading_df['Actual_Direction']).astype(int)
    signal_accuracy = trading_df['Signal_Correct'].mean()
    print(f"信号方向准确性: {signal_accuracy:.2%}")
    
    # 保存交易准备数据
    trading_df.to_csv('trading_signal_prep.csv', index=False)
    print("交易信号准备数据已保存到: trading_signal_prep.csv")
    
    # 显示前几行
    print("\n交易信号数据预览:")
    print(trading_df[['Date', 'Actual_Yield', 'Predicted_Yield', 'Signal', 'Up_Probability', 'Signal_Correct']].head())
    
    return trading_df, best_model_name

# 主函数
def main(data_file_path):
    """
    主流程函数
    """
    print("=" * 80)
    print("国债收益率预测模型系统")
    print("=" * 80)
    
    # 1. 读取数据
    print("\n1. 读取数据...")
    df = load_data(data_file_path)
    print(f"数据形状: {df.shape}")
    print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"10Y收益率统计:")
    print(df['10Y'].describe())
    
    # 2. 创建特征
    print("\n2. 创建特征...")
    X, y_reg, y_cls, df_features = create_features(df, lookback_days=20)
    print(f"特征形状: {X.shape}")
    print(f"回归目标形状: {y_reg.shape}")
    print(f"分类目标形状: {y_cls.shape}")
    
    # 3. 训练模型
    print("\n3. 训练和评估模型...")
    model_results, X_test, y_test = build_and_train_models(X, y_reg)
    
    if not model_results:
        print("没有模型成功训练，请检查数据或模型参数")
        return None
    
    # 4. 保存预测结果
    print("\n4. 保存预测结果...")
    results_df = save_predictions(model_results)
    
    # 5. 生成评价报告
    print("\n5. 生成模型评价报告...")
    report_df = generate_evaluation_report(model_results)
    
    # 6. 为交易信号准备数据
    print("\n6. 为交易信号准备数据...")
    trading_df, best_model = prepare_for_trading_signals(model_results, results_df)
    
    print("\n" + "=" * 80)
    print("国债收益率预测完成！")
    print("=" * 80)
    
    return {
        '原始数据': df,
        '特征数据': df_features,
        '模型结果': model_results,
        '预测结果': results_df,
        '评价报告': report_df,
        '交易准备数据': trading_df,
        '最佳模型': best_model
    }


# 如果有数据文件，运行主函数
if __name__ == "__main__":
    # 请将下面的文件路径替换为您的实际数据文件路径
    # 如果文件在当前目录，可以直接写文件名
    # 如果在其他目录，请写完整路径
    data_file = "CN_Yield_2018_2025.xlsx"
    
    try:
        print(f"尝试读取数据文件: {data_file}")
        results = main(data_file)
        
        if results:
            print("\n关键输出文件:")
            print("1. yield_predictions.csv - 所有模型的预测结果")
            print("2. model_evaluation_report.csv - 模型评价报告")
            print("3. trading_signal_prep.csv - 交易信号准备数据")
            
            print("\n下一步建议:")
            print("使用 trading_signal_prep.csv 中的预测数据和信号来实施仓位策略")
            print("可以根据报告中提到的策略进行选择:")
            print("  - 二值全仓策略: 基于Signal直接满仓做多(1)或做空(-1)")
            print("  - 带门槛的全仓策略: 只在Up_Probability > 0.55时做多, < 0.45时做空")
            print("  - 逐步加仓策略: 根据Up_Probability分阶段调整仓位")
            print("  - 连续型策略: 使用Sigmoid、Atanh等函数将Up_Probability映射到仓位")
        
    except FileNotFoundError:
        print(f"错误: 未找到数据文件 '{data_file}'")
        print("请确保:")
        print("1. 文件在当前目录或指定路径")
        print("2. 文件名正确")
        print("3. 文件格式为.xlsx")
        print("\n您可以尝试:")
        print("1. 检查文件路径")
        print("2. 修改代码中的data_file变量为正确的文件路径")
        print("3. 确保文件没有被其他程序打开")
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")
        print("\n调试建议:")
        print("1. 检查数据文件格式")
        print("2. 检查数据是否包含10年期国债收益率")
        print("3. 尝试减少lookback_days参数值")
        print("4. 检查是否有足够的数据点")
