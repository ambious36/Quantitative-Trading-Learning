import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid函数
from scipy.stats import norm
from scipy.special import atanh
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class BondTradingStrategy:
    def __init__(self, data_path):
        """
        初始化策略类
        Args:
            data_path: 数据文件路径
        """
        self.data = pd.read_csv(data_path, parse_dates=['date'])
        self.data.set_index('date', inplace=True)
        self.original_col = 'real_yield'  # 假设第二列是真实收益率
        self.pred_cols = [f'pred_{i}' for i in range(1, 9)]  # 8个预测列
        
        # 基础计算
        self.data['price_return'] = -self.data[self.original_col].diff()  # 价格收益率
        self.data = self.data.dropna()
        
        # 存储结果
        self.results = {}
        self.strategies = {}
        
    def generate_signals(self, model_idx=1, a=10):
        """
        生成交易信号
        Args:
            model_idx: 模型编号 (1-8)
            a: sigmoid函数的缩放参数
        """
        pred_col = f'pred_{model_idx}'
        
        # 二值信号
        self.data[f'binary_signal_m{model_idx}'] = np.where(
            self.data[pred_col] <= self.data[self.original_col].shift(1),
            1, 0
        )
        
        # 连续信号
        diff = self.data[self.original_col].shift(1) - self.data[pred_col]
        self.data[f'continuous_signal_m{model_idx}'] = 1 / (1 + np.exp(-a * diff))
        
        return self.data
    
    # ========== 仓位策略 ==========
    
    def binary_full_position(self, signal_col, threshold=0.5):
        """
        多空全仓策略（基准策略）
        Args:
            signal_col: 信号列名
            threshold: 阈值，默认0.5
        """
        position = np.where(self.data[signal_col] > threshold, 1, 
                           np.where(self.data[signal_col] < threshold, -1, 0))
        return position
    
    def threshold_position(self, signal_col, threshold_low=0.45, threshold_high=0.55):
        """
        带门槛的全仓策略
        Args:
            signal_col: 信号列名
            threshold_low: 下限阈值
            threshold_high: 上限阈值
        """
        position = np.zeros(len(self.data))
        # 信号在模糊区间时平仓
        mask = (self.data[signal_col] >= threshold_low) & (self.data[signal_col] <= threshold_high)
        position[~mask] = np.where(self.data.loc[~mask, signal_col] > threshold_high, 1, -1)
        return position
    
    def gradual_position(self, signal_col, threshold_low=0.45, threshold_high=0.55, 
                         step_size=0.2, max_position=1):
        """
        逐步加仓策略
        Args:
            signal_col: 信号列名
            threshold_low: 下限阈值
            threshold_high: 上限阈值
            step_size: 每次调整的仓位幅度
            max_position: 最大仓位
        """
        position = np.zeros(len(self.data))
        current_pos = 0
        
        for i in range(1, len(self.data)):
            signal = self.data.iloc[i][signal_col]
            
            if signal > threshold_high:  # 强多头信号
                target_pos = max_position
            elif signal < threshold_low:  # 强空头信号
                target_pos = -max_position
            else:  # 模糊区间，保持仓位
                target_pos = current_pos
            
            # 逐步调整仓位
            if target_pos > current_pos:
                current_pos = min(current_pos + step_size, target_pos)
            elif target_pos < current_pos:
                current_pos = max(current_pos - step_size, target_pos)
            
            position[i] = current_pos
        
        return position
    
    # ========== 连续策略 ==========
    
    def linear_position(self, signal_col, k=1.0):
        """
        线性策略 (风险中性)
        Args:
            signal_col: 信号列名
            k: 缩放参数
        """
        # 将信号从[0,1]映射到[-1,1]
        x = self.data[signal_col] * 2 - 1  # 映射到[-1,1]
        position = k * x
        position = np.clip(position, -1, 1)  # 限制在[-1,1]之间
        return position
    
    def sigmoid_position(self, signal_col, a=5.0):
        """
        Sigmoid策略 (风险偏好型)
        Args:
            signal_col: 信号列名
            a: 缩放参数，控制曲线陡峭程度
        """
        x = self.data[signal_col] * 2 - 1  # 映射到[-1,1]
        position = 2 * (expit(a * x) - 0.5)
        return position
    
    def normal_position(self, signal_col, a=2.0):
        """
        正态策略 (风险偏好型)
        Args:
            signal_col: 信号列名
            a: 缩放参数
        """
        x = self.data[signal_col] * 2 - 1  # 映射到[-1,1]
        position = 2 * (norm.cdf(a * x) - 0.5)
        return position
    
    def atanh_position(self, signal_col, a=0.8):
        """
        Atanh策略 (风险厌恶型)
        Args:
            signal_col: 信号列名
            a: 缩放参数
        """
        x = self.data[signal_col] * 2 - 1  # 映射到[-1,1]
        # 限制在(-0.99, 0.99)之间，避免无穷大
        x_scaled = a * x
        x_scaled = np.clip(x_scaled, -0.99, 0.99)
        
        position = atanh(x_scaled) / atanh(a)
        position = np.clip(position, -1, 1)
        return position
    
    def atanh_sigmoid_position(self, signal_col, a_sigmoid=5.0, a_atanh=0.8):
        """
        Atanh-Sigmoid混合策略
        多头时使用Sigmoid，空头时使用Atanh
        Args:
            signal_col: 信号列名
            a_sigmoid: Sigmoid的缩放参数
            a_atanh: Atanh的缩放参数
        """
        position = np.zeros(len(self.data))
        
        # 计算多头信号（信号>0.5）
        mask_bull = self.data[signal_col] >= 0.5
        if mask_bull.any():
            x_bull = self.data.loc[mask_bull, signal_col] * 2 - 1
            position[mask_bull] = 2 * (expit(a_sigmoid * x_bull) - 0.5)
        
        # 计算空头信号（信号<0.5）
        mask_bear = self.data[signal_col] < 0.5
        if mask_bear.any():
            x_bear = self.data.loc[mask_bear, signal_col] * 2 - 1
            x_scaled = a_atanh * x_bear
            x_scaled = np.clip(x_scaled, -0.99, 0.99)
            position[mask_bear] = atanh(x_scaled) / atanh(a_atanh)
        
        return position
    
    # ========== 回测引擎 ==========
    
    def backtest(self, position, model_name, strategy_name, 
                 rf_rate=0.0, transaction_cost=0.0001):
        """
        执行回测
        Args:
            position: 仓位序列
            model_name: 模型名称
            strategy_name: 策略名称
            rf_rate: 无风险利率
            transaction_cost: 交易成本（双边）
        """
        # 计算每日收益
        daily_return = position * self.data['price_return']
        
        # 考虑保证金外的资金收益（无风险利率）
        unused_capital_return = (1 - np.abs(position)) * rf_rate / 252
        
        # 计算总收益
        total_daily_return = daily_return + unused_capital_return
        
        # 计算换手率
        position_change = np.abs(np.diff(position, prepend=0))
        turnover = position_change.mean() * 252  # 年化换手率
        
        # 计算交易成本
        transaction_cost_total = position_change * transaction_cost
        net_return = total_daily_return - transaction_cost_total
        
        # 计算净值
        net_value = (1 + net_return).cumprod()
        
        # 计算最大回撤
        rolling_max = net_value.expanding().max()
        drawdown = (net_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 计算年化收益率
        total_days = len(net_value)
        annual_return = (net_value.iloc[-1] ** (252/total_days) - 1) 
        
        # 计算夏普比率（假设无风险利率为0）
        excess_return = net_return - rf_rate/252
        sharpe_ratio = np.sqrt(252) * excess_return.mean() / excess_return.std() if excess_return.std() > 0 else 0
        
        # 计算卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 存储结果
        key = f"{model_name}_{strategy_name}"
        self.results[key] = {
            'net_value': net_value,
            'position': position,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'turnover': turnover,
            'daily_return': net_return
        }
        
        print(f"{key}: 年化收益={annual_return*100:.2f}%, "
              f"最大回撤={max_drawdown*100:.2f}%, "
              f"夏普={sharpe_ratio:.2f}, "
              f"卡玛={calmar_ratio:.2f}, "
              f"换手率={turnover*100:.1f}%")
        
        return self.results[key]
    
    # ========== 批量回测 ==========
    
    def run_all_strategies(self, model_indices=None):
        """
        运行所有策略
        Args:
            model_indices: 模型索引列表，默认为[1,2,3,4,5,6,7,8]
        """
        if model_indices is None:
            model_indices = [1, 2, 3, 4, 5, 6, 7, 8]
        
        all_results = {}
        
        for model_idx in model_indices:
            # 生成信号
            self.generate_signals(model_idx)
            
            model_name = f"Model_{model_idx}"
            print(f"\n=== 测试模型: {model_name} ===")
            
            # 信号列名
            binary_signal_col = f'binary_signal_m{model_idx}'
            cont_signal_col = f'continuous_signal_m{model_idx}'
            
            # 1. 多空全仓策略（基准）
            position = self.binary_full_position(binary_signal_col)
            self.backtest(position, model_name, "binary_full")
            
            # 2. 带门槛的全仓策略
            position = self.threshold_position(binary_signal_col)
            self.backtest(position, model_name, "threshold_full")
            
            # 3. 逐步加仓策略
            position = self.gradual_position(binary_signal_col)
            self.backtest(position, model_name, "gradual")
            
            # 4. 线性策略
            position = self.linear_position(cont_signal_col)
            self.backtest(position, model_name, "linear")
            
            # 5. Sigmoid策略
            position = self.sigmoid_position(cont_signal_col)
            self.backtest(position, model_name, "sigmoid")
            
            # 6. 正态策略
            position = self.normal_position(cont_signal_col)
            self.backtest(position, model_name, "normal")
            
            # 7. Atanh策略
            position = self.atanh_position(cont_signal_col)
            self.backtest(position, model_name, "atanh")
            
            # 8. Atanh-Sigmoid策略
            position = self.atanh_sigmoid_position(cont_signal_col)
            self.backtest(position, model_name, "atanh_sigmoid")
        
        return self.results
    
    # ========== 可视化 ==========
    
    def plot_net_value_comparison(self, strategies_to_plot=None, model_idx=1):
        """
        绘制净值曲线对比
        Args:
            strategies_to_plot: 要绘制的策略列表
            model_idx: 模型索引
        """
        if strategies_to_plot is None:
            strategies_to_plot = ['binary_full', 'threshold_full', 'gradual', 
                                 'linear', 'sigmoid', 'atanh']
        
        plt.figure(figsize=(12, 6))
        model_name = f"Model_{model_idx}"
        
        for strategy in strategies_to_plot:
            key = f"{model_name}_{strategy}"
            if key in self.results:
                plt.plot(self.results[key]['net_value'], label=f'{strategy}')
        
        plt.title(f'净值曲线对比 - {model_name}')
        plt.xlabel('日期')
        plt.ylabel('净值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_position_comparison(self, strategies_to_plot=None, model_idx=1, 
                                 start_date=None, end_date=None):
        """
        绘制仓位对比
        Args:
            strategies_to_plot: 要绘制的策略列表
            model_idx: 模型索引
            start_date, end_date: 时间范围
        """
        if strategies_to_plot is None:
            strategies_to_plot = ['binary_full', 'gradual', 'sigmoid', 'atanh']
        
        model_name = f"Model_{model_idx}"
        n_strategies = len(strategies_to_plot)
        
        fig, axes = plt.subplots(n_strategies, 1, figsize=(12, 3*n_strategies), sharex=True)
        
        if n_strategies == 1:
            axes = [axes]
        
        for i, strategy in enumerate(strategies_to_plot):
            key = f"{model_name}_{strategy}"
            if key in self.results:
                position_data = self.results[key]['position']
                
                # 截取指定时间段
                if start_date is not None and end_date is not None:
                    mask = (self.data.index >= start_date) & (self.data.index <= end_date)
                    position_data = position_data[mask]
                    dates = self.data.index[mask]
                else:
                    dates = self.data.index
                    # 只显示最近的数据
                    if len(dates) > 100:
                        dates = dates[-100:]
                        position_data = position_data[-100:]
                
                axes[i].plot(dates, position_data, label=f'{strategy}', linewidth=1)
                axes[i].fill_between(dates, 0, position_data, 
                                     where=position_data>0, 
                                     color='red', alpha=0.3)
                axes[i].fill_between(dates, 0, position_data, 
                                     where=position_data<0, 
                                     color='green', alpha=0.3)
                axes[i].set_ylabel('仓位')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.xlabel('日期')
        plt.suptitle(f'仓位对比 - {model_name}')
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_comparison(self, strategies_to_plot=None, model_idx=1):
        """
        绘制回撤曲线对比
        Args:
            strategies_to_plot: 要绘制的策略列表
            model_idx: 模型索引
        """
        if strategies_to_plot is None:
            strategies_to_plot = ['binary_full', 'threshold_full', 'sigmoid', 'atanh']
        
        plt.figure(figsize=(12, 6))
        model_name = f"Model_{model_idx}"
        
        for strategy in strategies_to_plot:
            key = f"{model_name}_{strategy}"
            if key in self.results:
                net_value = self.results[key]['net_value']
                rolling_max = net_value.expanding().max()
                drawdown = (net_value - rolling_max) / rolling_max
                plt.plot(drawdown, label=f'{strategy}', linewidth=1)
        
        plt.title(f'回撤曲线对比 - {model_name}')
        plt.xlabel('日期')
        plt.ylabel('回撤')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.show()
    
    def plot_strategy_performance_matrix(self, metric='annual_return'):
        """
        绘制策略表现矩阵热力图
        Args:
            metric: 评价指标 ('annual_return', 'sharpe_ratio', 'calmar_ratio', 'max_drawdown')
        """
        # 提取所有策略结果
        performance_data = []
        model_names = []
        strategy_names = set()
        
        for key, result in self.results.items():
            model_name = key.split('_')[0] + '_' + key.split('_')[1]
            strategy_name = key.split('_', 2)[2]
            
            if model_name not in model_names:
                model_names.append(model_name)
            
            strategy_names.add(strategy_name)
            
            if metric == 'annual_return':
                value = result['annual_return']
            elif metric == 'sharpe_ratio':
                value = result['sharpe_ratio']
            elif metric == 'calmar_ratio':
                value = result['calmar_ratio']
            elif metric == 'max_drawdown':
                value = result['max_drawdown'] * 100  # 转换为百分比
        
        # 转换为DataFrame
        strategy_names = sorted(list(strategy_names))
        df = pd.DataFrame(index=model_names, columns=strategy_names)
        
        for key, result in self.results.items():
            model_name = key.split('_')[0] + '_' + key.split('_')[1]
            strategy_name = key.split('_', 2)[2]
            
            if metric == 'annual_return':
                value = result['annual_return']
            elif metric == 'sharpe_ratio':
                value = result['sharpe_ratio']
            elif metric == 'calmar_ratio':
                value = result['calmar_ratio']
            elif metric == 'max_drawdown':
                value = result['max_drawdown'] * 100
            
            df.loc[model_name, strategy_name] = value
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        plt.imshow(df.astype(float), cmap='RdYlGn', aspect='auto')
        plt.colorbar(label=metric)
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=45)
        plt.yticks(range(len(model_names)), model_names)
        plt.title(f'策略表现矩阵 - {metric}')
        plt.tight_layout()
        plt.show()
        
        return df
    def export_performance_metrics_to_csv(self, output_dir='./performance_metrics/'):
        """
        导出各模型、各策略的性能指标到CSV文件
        
        Args:
            output_dir: 输出目录路径
        """
        import os
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 收集所有模型和策略名称
        models = set()
        strategies = set()
        
        for key in self.results.keys():
            parts = key.split('_', 2)
            if len(parts) >= 3:
                model_name = f"{parts[0]}_{parts[1]}"
                strategy_name = parts[2]
                models.add(model_name)
                strategies.add(strategy_name)
        
        # 排序
        models = sorted(list(models))
        strategies = sorted(list(strategies))
        
        # 初始化指标DataFrame
        df_annual_return = pd.DataFrame(index=strategies, columns=models)
        df_max_drawdown = pd.DataFrame(index=strategies, columns=models)
        df_sharpe_ratio = pd.DataFrame(index=strategies, columns=models)
        df_calmar_ratio = pd.DataFrame(index=strategies, columns=models)
        df_turnover = pd.DataFrame(index=strategies, columns=models)
        
        # 填充数据
        for key, result in self.results.items():
            parts = key.split('_', 2)
            if len(parts) >= 3:
                model_name = f"{parts[0]}_{parts[1]}"
                strategy_name = parts[2]
                
                df_annual_return.loc[strategy_name, model_name] = result['annual_return'] * 100
                df_max_drawdown.loc[strategy_name, model_name] = result['max_drawdown'] * 100  # 转换为百分比
                df_sharpe_ratio.loc[strategy_name, model_name] = result['sharpe_ratio']
                df_calmar_ratio.loc[strategy_name, model_name] = result['calmar_ratio']
                df_turnover.loc[strategy_name, model_name] = result['turnover'] * 100  # 转换为百分比
        
        # 保存到CSV文件
        df_annual_return.to_csv(os.path.join(output_dir, 'annual_return.csv'))
        df_max_drawdown.to_csv(os.path.join(output_dir, 'max_drawdown.csv'))
        df_sharpe_ratio.to_csv(os.path.join(output_dir, 'sharpe_ratio.csv'))
        df_calmar_ratio.to_csv(os.path.join(output_dir, 'calmar_ratio.csv'))
        df_turnover.to_csv(os.path.join(output_dir, 'turnover.csv'))
        
        print(f"性能指标已导出到目录: {output_dir}")
        print(f"文件列表:")
        print(f"1. annual_return.csv")
        print(f"2. max_drawdown.csv")
        print(f"3. sharpe_ratio.csv")
        print(f"4. calmar_ratio.csv")
        print(f"5. turnover.csv")
        
        return {
            'annual_return': df_annual_return,
            'max_drawdown': df_max_drawdown,
            'sharpe_ratio': df_sharpe_ratio,
            'calmar_ratio': df_calmar_ratio,
            'turnover': df_turnover
        }

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 创建策略实例（假设数据文件名为 bond_yield_data.csv）
    # 注意：您的数据文件需要包含以下列：date, real_yield, pred_1, pred_2, ..., pred_8
    strategy = BondTradingStrategy("bond_yield_data.csv")
    
    # 2. 运行所有策略（可以指定模型，如只测试前4个模型）
    print("开始回测...")
    results = strategy.run_all_strategies(model_indices=[1, 2, 3, 4])
    
    # 3. 可视化
    # 3.1 净值曲线对比
    strategy.plot_net_value_comparison(model_idx=1)
    
    # 3.2 仓位对比（最近100天）
    strategy.plot_position_comparison(model_idx=1)
    
    # 3.3 回撤曲线对比
    strategy.plot_drawdown_comparison(model_idx=1)
    
    # 3.4 策略表现矩阵
    df_annual_return = strategy.plot_strategy_performance_matrix('annual_return')
    df_sharpe = strategy.plot_strategy_performance_matrix('sharpe_ratio')
    
    print("\n策略对比完成！")