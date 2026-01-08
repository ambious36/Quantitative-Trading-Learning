import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ChinaTreasuryPCAAnalysis:
    """
    使用PCA分析中国国债利率期限结构
    """
    
    def __init__(self, data_path, n_components=3):
        """
        初始化
        
        Parameters
        ----------
        data_path : str
            数据文件路径
        n_components : int
            PCA主成分数量，通常为3（水平、斜率、曲率）
        """
        self.data_path = data_path
        self.n_components = n_components
        self.pca_model = None
        self.factors = None
        self.loadings = None
        
        # 读取数据
        self.load_data()
    
    def load_data(self):
        """加载国债收益率数据"""
        print("加载中国国债收益率数据...")
        self.data = pd.read_excel(self.data_path)
        
        # 确保日期列为datetime格式
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
        elif 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        
        # 检查数据
        print(f"数据时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        print(f"数据频率: {pd.infer_freq(self.data.index)}")
        print(f"数据维度: {self.data.shape}")
        print(f"可用期限: {list(self.data.columns)}")
        
        # 转换百分比为小数（如果数据是百分比形式）
        print("\n数据转换检查...")
        for col in self.data.columns:
            col_mean = self.data[col].mean()
            print(f"{col}: 均值 = {col_mean:.4f}")
            
            if col_mean > 1:  # 如果均值大于1，假设是百分比形式
                print(f"  → 将{col}从百分比转换为小数（除以100）")
                self.data[col] = self.data[col] / 100
        
        # 处理缺失值
        self.data = self.data.ffill()
        
        # 保存原始数据
        self.original_data = self.data.copy()
        
        # 计算期限数值
        self.maturities_numeric = self.get_maturity_numeric(self.data.columns.tolist())
        print(f"数值期限: {self.maturities_numeric}")
    
    def get_maturity_numeric(self, maturities):
        """将期限字符串转换为数值（年）"""
        numeric_maturities = []
        for m in maturities:
            if 'M' in m:
                months = float(m.replace('M', ''))
                numeric_maturities.append(months / 12)
            elif 'Y' in m:
                years = float(m.replace('Y', ''))
                numeric_maturities.append(years)
            else:
                try:
                    numeric_maturities.append(float(m))
                except:
                    numeric_maturities.append(1.0)
        return np.array(numeric_maturities)
    
    def fit_pca_model(self):
        """拟合PCA模型"""
        print(f"\n拟合PCA模型 (主成分数={self.n_components})...")
        
        # 标准化数据（去除均值）
        self.data_mean = self.data.mean()
        self.data_std = self.data.std()
        self.data_normalized = (self.data - self.data_mean) / self.data_std
        
        # 应用PCA
        self.pca_model = PCA(n_components=self.n_components)
        self.factors = self.pca_model.fit_transform(self.data_normalized.values)
        self.loadings = self.pca_model.components_
        
        # 计算解释方差比例
        explained_variance_ratio = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"主成分解释方差比例: {explained_variance_ratio}")
        print(f"累计解释方差比例: {cumulative_variance}")
        print(f"总解释方差: {cumulative_variance[-1]:.2%}")
        
        # 创建因子时间序列DataFrame
        factor_names = [f'PC{i+1}' for i in range(self.n_components)]
        self.factors_df = pd.DataFrame(
            self.factors,
            index=self.data.index,
            columns=factor_names
        )
        
        # 为因子赋予经济含义
        self.interpret_factors()
        
        return self.factors_df
    
    def interpret_factors(self):
        """解释PCA因子的经济含义"""
        print("\nPCA因子经济含义解释:")
        
        for i in range(self.n_components):
            loading = self.loadings[i]
            
            # 计算各期限的因子载荷
            loading_by_maturity = dict(zip(self.data.columns, loading))
            
            # 判断因子类型
            if i == 0:
                # PC1通常代表水平因子（所有期限同向变化）
                mean_loading = np.mean(loading)
                if mean_loading > 0:
                    print(f"PC1 (水平因子): 所有期限同向变化，解释方差{self.pca_model.explained_variance_ratio_[i]:.2%}")
                else:
                    print(f"PC1 (水平因子): 所有期限反向变化，解释方差{self.pca_model.explained_variance_ratio_[i]:.2%}")
                
            elif i == 1:
                # PC2通常代表斜率因子（短期和长期反向变化）
                # 计算短期和长期载荷的差异
                short_term_cols = [col for col in self.data.columns if 'M' in col or ('Y' in col and float(col.replace('Y', '')) <= 2)]
                long_term_cols = [col for col in self.data.columns if '10Y' in col or '30Y' in col]
                
                short_loading = np.mean([loading_by_maturity[col] for col in short_term_cols if col in loading_by_maturity])
                long_loading = np.mean([loading_by_maturity[col] for col in long_term_cols if col in loading_by_maturity])
                
                if short_loading * long_loading < 0:  # 符号相反
                    print(f"PC2 (斜率因子): 短期{short_loading:.3f} vs 长期{long_loading:.3f}，解释方差{self.pca_model.explained_variance_ratio_[i]:.2%}")
                else:
                    print(f"PC2 (可能为曲率因子): 解释方差{self.pca_model.explained_variance_ratio_[i]:.2%}")
                    
            elif i == 2:
                # PC3通常代表曲率因子（中期与两端反向变化）
                print(f"PC3 (曲率因子): 解释方差{self.pca_model.explained_variance_ratio_[i]:.2%}")
        
        # 绘制因子载荷图
        self.plot_factor_loadings()
    
    def plot_factor_loadings(self):
        """绘制因子载荷图"""
        fig, axes = plt.subplots(1, self.n_components, figsize=(4*self.n_components, 4))
        
        if self.n_components == 1:
            axes = [axes]
        
        for i in range(self.n_components):
            ax = axes[i]
            loadings = self.loadings[i]
            
            # 按期限排序
            sorted_indices = np.argsort(self.maturities_numeric)
            sorted_maturities = self.maturities_numeric[sorted_indices]
            sorted_loadings = loadings[sorted_indices]
            sorted_labels = np.array(self.data.columns)[sorted_indices]
            
            ax.plot(sorted_maturities, sorted_loadings, 'b-o', linewidth=2, markersize=6)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title(f'PC{i+1} 载荷 (解释方差{self.pca_model.explained_variance_ratio_[i]:.2%})')
            ax.set_xlabel('期限（年）')
            ax.set_ylabel('载荷')
            ax.grid(True, alpha=0.3)
            
            # 标记关键期限点
            for tau, loading, label in zip(sorted_maturities, sorted_loadings, sorted_labels):
                ax.annotate(label, (tau, loading), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def reconstruct_yields(self, factors=None):
        """使用因子重建收益率"""
        if factors is None:
            factors = self.factors
        
        # 重建标准化数据
        reconstructed_normalized = self.pca_model.inverse_transform(factors)
        
        # 反标准化
        reconstructed = reconstructed_normalized * self.data_std.values + self.data_mean.values
        
        return pd.DataFrame(reconstructed, 
                          index=self.data.index, 
                          columns=self.data.columns)
    
    def plot_factors_timeseries(self):
        """绘制因子时间序列"""
        fig, axes = plt.subplots(self.n_components, 1, figsize=(12, 3*self.n_components))
        
        if self.n_components == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            factor_name = f'PC{i+1}'
            factor_data = self.factors_df[factor_name]
            
            ax.plot(factor_data.index, factor_data, 'b-', linewidth=1.5)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=factor_data.mean(), color='g', linestyle='-', alpha=0.7)
            ax.fill_between(factor_data.index, 
                          factor_data.mean() - factor_data.std(),
                          factor_data.mean() + factor_data.std(),
                          alpha=0.2, color='blue')
            
            ax.set_title(f'{factor_name} 时间序列 (均值={factor_data.mean():.4f}, 标准差={factor_data.std():.4f})')
            ax.set_xlabel('日期')
            ax.set_ylabel('因子值')
            ax.grid(True, alpha=0.3)
            ax.legend([factor_name, '零线', f'均值={factor_data.mean():.4f}'], 
                     loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_fitting_accuracy(self, date=None):
        """分析拟合精度"""
        if date is None:
            date = self.data.index[-1]
        
        # 重建收益率
        reconstructed = self.reconstruct_yields()
        
        # 计算误差
        actual_yields = self.original_data.loc[date]
        reconstructed_yields = reconstructed.loc[date]
        errors = actual_yields - reconstructed_yields
        
        # 计算统计量
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        max_error = np.max(np.abs(errors))
        
        # 绘制拟合对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：收益率曲线对比
        ax1.plot(self.maturities_numeric, actual_yields.values, 'ro-', 
                linewidth=2, markersize=8, label='实际收益率')
        ax1.plot(self.maturities_numeric, reconstructed_yields.values, 'b--s', 
                linewidth=2, markersize=6, label='PCA重建')
        
        # 连接误差线
        for tau, y_act, y_rec in zip(self.maturities_numeric, 
                                    actual_yields.values, 
                                    reconstructed_yields.values):
            ax1.plot([tau, tau], [y_act, y_rec], 'gray', linestyle=':', alpha=0.5)
        
        ax1.set_title(f'PCA收益率曲线拟合 - {date.strftime("%Y-%m-%d")}\nRMSE={rmse:.4%} ({rmse*10000:.1f}bp)')
        ax1.set_xlabel('期限（年）')
        ax1.set_ylabel('收益率')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 标注关键点
        for i, (tau, y_act, y_rec) in enumerate(zip(self.maturities_numeric,
                                                   actual_yields.values,
                                                   reconstructed_yields.values)):
            error_bp = (y_act - y_rec) * 10000
            ax1.annotate(f'{error_bp:.1f}bp', (tau, (y_act+y_rec)/2),
                        textcoords="offset points", xytext=(0,0),
                        ha='center', fontsize=8, color='gray')
        
        # 右图：拟合误差分布
        colors = ['red' if e > 0 else 'blue' for e in errors]
        bars = ax2.bar(range(len(errors)), errors * 10000, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax2.set_title(f'拟合误差分布 (MAE={mae*10000:.1f}bp, Max={max_error*10000:.1f}bp)')
        ax2.set_xlabel('期限点')
        ax2.set_ylabel('误差（基点）')
        ax2.set_xticks(range(len(errors)))
        ax2.set_xticklabels(actual_yields.index, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加误差值标注
        for i, (bar, error) in enumerate(zip(bars, errors)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    height + (1 if height > 0 else -5),
                    f'{error*10000:.1f}', 
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细统计
        print(f"\n详细拟合误差分析 ({date.strftime('%Y-%m-%d')}):")
        print("="*60)
        print(f"{'期限':<8} {'实际值':<10} {'重建值':<10} {'误差(bp)':<12} {'相对误差':<12}")
        print("-"*60)
        
        for maturity, y_act, y_rec, error in zip(actual_yields.index,
                                                actual_yields.values,
                                                reconstructed_yields.values,
                                                errors):
            rel_error = abs(error/y_act) if y_act != 0 else np.nan
            print(f"{maturity:<8} {y_act:>8.4%} {y_rec:>8.4%} {error*10000:>10.1f} {rel_error:>11.2%}")
        
        print("-"*60)
        print(f"RMSE: {rmse:.4%} ({rmse*10000:.1f} bp)")
        print(f"MAE: {mae:.4%} ({mae*10000:.1f} bp)")
        print(f"最大绝对误差: {max_error:.4%} ({max_error*10000:.1f} bp)")
        
        return rmse, mae, max_error
    
    def calculate_overall_fitting_metrics(self):
        """计算整体拟合指标"""
        print("\n计算整体拟合质量...")
        
        reconstructed = self.reconstruct_yields()
        
        all_rmse = []
        all_mae = []
        all_max_error = []
        
        for date in self.data.index:
            actual = self.original_data.loc[date]
            recon = reconstructed.loc[date]
            errors = actual - recon
            
            rmse = np.sqrt(np.mean(errors**2))
            mae = np.mean(np.abs(errors))
            max_error = np.max(np.abs(errors))
            
            all_rmse.append(rmse)
            all_mae.append(mae)
            all_max_error.append(max_error)
        
        avg_rmse = np.mean(all_rmse)
        avg_mae = np.mean(all_mae)
        avg_max_error = np.mean(all_max_error)
        
        print(f"平均RMSE: {avg_rmse:.4%} ({avg_rmse*10000:.1f} bp)")
        print(f"平均MAE: {avg_mae:.4%} ({avg_mae*10000:.1f} bp)")
        print(f"平均最大误差: {avg_max_error:.4%} ({avg_max_error*10000:.1f} bp)")
        print(f"RMSE范围: {np.min(all_rmse)*10000:.1f} - {np.max(all_rmse)*10000:.1f} bp")
        
        return {
            'avg_rmse': avg_rmse,
            'avg_mae': avg_mae,
            'avg_max_error': avg_max_error,
            'all_rmse': all_rmse,
            'all_mae': all_mae,
            'all_max_error': all_max_error
        }
    
    def relative_value_analysis_pca(self, target_maturity='10Y', window_fast=5, window_slow=40):
        """
        PCA相对价值交易分析
        
        Parameters
        ----------
        target_maturity : str
            目标期限
        window_fast : int
            快速移动平均窗口
        window_slow : int
            慢速移动平均窗口
        """
        print(f"\nPCA相对价值分析: {target_maturity}收益率")
        
        if target_maturity not in self.data.columns:
            print(f"目标期限 {target_maturity} 不在数据中")
            return
        
        # 计算每个时间点的拟合误差
        fitting_errors_list = []
        
        reconstructed = self.reconstruct_yields()
        
        for date in self.data.index:
            actual_rate = self.original_data.loc[date, target_maturity]
            recon_rate = reconstructed.loc[date, target_maturity]
            error = actual_rate - recon_rate
            fitting_errors_list.append(error)
        
        fitting_errors = np.array(fitting_errors_list)
        
        if len(fitting_errors) < window_slow * 2:
            print(f"数据点不足，需要至少 {window_slow*2} 个数据点")
            return
        
        # 计算移动平均信号
        ma_fast = pd.Series(fitting_errors).rolling(window=window_fast, min_periods=1).mean()
        ma_slow = pd.Series(fitting_errors).rolling(window=window_slow, min_periods=1).mean()
        signal = ma_fast - ma_slow
        
        # 去中心化的拟合误差
        demeaned_errors = fitting_errors - np.mean(fitting_errors)
        
        # 设置交易阈值（基于误差和信号的波动率）
        error_threshold = np.std(demeaned_errors) * 0.5
        signal_threshold = np.std(signal) * 0.3
        
        # 识别交易信号
        trade_signals = []
        trade_dates = []
        trade_details = []
        
        for i in range(window_slow, len(demeaned_errors)):
            error_condition = abs(demeaned_errors[i]) > error_threshold
            signal_condition = abs(signal[i]) > signal_threshold
            
            if error_condition and signal_condition:
                if demeaned_errors[i] > 0 and signal[i] < -signal_threshold:
                    trade_signals.append(1)  # 买入信号（实际利率高于PCA重建）
                    trade_type = "买入"
                    reason = f"实际利率比PCA预测高{demeaned_errors[i]*10000:.1f}bp"
                elif demeaned_errors[i] < 0 and signal[i] > signal_threshold:
                    trade_signals.append(-1)  # 卖出信号（实际利率低于PCA重建）
                    trade_type = "卖出"
                    reason = f"实际利率比PCA预测低{abs(demeaned_errors[i])*10000:.1f}bp"
                else:
                    trade_signals.append(0)
                    trade_type = None
                    reason = None
                
                if trade_type:
                    trade_date = self.data.index[i]
                    trade_dates.append(trade_date)
                    trade_details.append({
                        'date': trade_date,
                        'type': trade_type,
                        'actual_rate': self.original_data.loc[trade_date, target_maturity],
                        'pca_rate': reconstructed.loc[trade_date, target_maturity],
                        'error_bp': demeaned_errors[i] * 10000,
                        'signal': signal[i] * 10000,
                        'reason': reason
                    })
            else:
                trade_signals.append(0)
        
        # 绘制结果
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. 拟合误差时间序列
        axes[0].plot(self.data.index, demeaned_errors*10000, 'b-', linewidth=1.5, alpha=0.7)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].axhline(y=error_threshold*10000, color='g', linestyle=':', alpha=0.5)
        axes[0].axhline(y=-error_threshold*10000, color='g', linestyle=':', alpha=0.5)
        axes[0].fill_between(self.data.index, 
                           -error_threshold*10000, 
                           error_threshold*10000, 
                           alpha=0.1, color='gray')
        axes[0].set_title(f'{target_maturity} PCA拟合误差 (阈值=±{error_threshold*10000:.1f}bp)', 
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('误差（基点）')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(['拟合误差', '零线', f'阈值±{error_threshold*10000:.1f}bp'])
        
        # 2. 交易信号
        axes[1].plot(self.data.index, signal*10000, 'g-', linewidth=1.5, alpha=0.7, label='信号')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=signal_threshold*10000, color='orange', linestyle=':', alpha=0.5)
        axes[1].axhline(y=-signal_threshold*10000, color='orange', linestyle=':', alpha=0.5)
        axes[1].fill_between(self.data.index, 
                           -signal_threshold*10000, 
                           signal_threshold*10000, 
                           alpha=0.1, color='gray')
        axes[1].set_title(f'交易信号 (阈值=±{signal_threshold*10000:.1f}bp)', 
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('信号强度（基点）')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(['信号', '零线', f'阈值±{signal_threshold*10000:.1f}bp'])
        
        # 3. 交易信号点
        axes[2].plot(self.data.index, np.zeros_like(demeaned_errors), 'k-', alpha=0.3)
        
        # 标记交易点
        buy_dates = [td['date'] for td in trade_details if td['type'] == '买入']
        sell_dates = [td['date'] for td in trade_details if td['type'] == '卖出']
        
        if buy_dates:
            axes[2].scatter(buy_dates, [0]*len(buy_dates), 
                          color='green', s=100, marker='^', label='买入信号', zorder=5)
        
        if sell_dates:
            axes[2].scatter(sell_dates, [0]*len(sell_dates), 
                          color='red', s=100, marker='v', label='卖出信号', zorder=5)
        
        axes[2].set_title(f'交易信号点 (总信号数: {len(trade_details)})', 
                         fontsize=12, fontweight='bold')
        axes[2].set_xlabel('日期')
        axes[2].set_ylabel('交易信号')
        axes[2].set_ylim([-1, 1])
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # 打印交易统计
        print(f"\nPCA交易策略统计 ({target_maturity}):")
        print("="*60)
        print(f"总交易日数: {len(self.data)}")
        print(f"交易信号数: {len(trade_details)}")
        print(f"买入信号: {len(buy_dates)}")
        print(f"卖出信号: {len(sell_dates)}")
        print(f"信号频率: {len(trade_details)/len(self.data):.2%}")
        
        if trade_details:
            print(f"\n最新交易信号:")
            for i, trade in enumerate(trade_details[-5:]):  # 显示最近5个信号
                print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['type']} "
                      f"(实际={trade['actual_rate']:.4%}, PCA={trade['pca_rate']:.4%}, "
                      f"误差={trade['error_bp']:.1f}bp, 信号={trade['signal']:.1f}bp)")
        
        # 策略回测
        if len(trade_details) >= 2:
            returns = []
            for i in range(len(trade_details)-1):
                trade = trade_details[i]
                next_trade = trade_details[i+1]
                
                # 计算交易收益（简化：假设收益率回归到PCA预测值）
                if trade['type'] == '买入':
                    # 买入后期望实际利率下降（向PCA预测值回归）
                    price_change = -trade['error_bp']  # 误差缩小导致价格上升
                else:  # 卖出
                    # 卖出后期望实际利率上升（向PCA预测值回归）
                    price_change = trade['error_bp']  # 误差扩大导致价格下降
                
                returns.append(price_change)
            
            if returns:
                avg_return = np.mean(returns)
                win_rate = sum([1 for r in returns if r > 0]) / len(returns)
                sharpe_ratio = avg_return / np.std(returns) * np.sqrt(252/30) if np.std(returns) > 0 else 0
                
                print(f"\n策略回测结果:")
                print(f"  总交易次数: {len(returns)}")
                print(f"  平均收益: {avg_return:.1f} bp/次")
                print(f"  胜率: {win_rate:.2%}")
                print(f"  夏普比率: {sharpe_ratio:.2f}")
                print(f"  平均持有期: {len(self.data)/len(trade_details):.1f} 天")
        
        return trade_details
    
    def plot_historical_yield_curves_pca(self, n_dates=12):
        """绘制历史收益率曲线PCA重建对比"""
        # 选择等间隔的日期
        if len(self.data) > n_dates:
            step = len(self.data) // n_dates
            selected_indices = range(0, len(self.data), step)[:n_dates]
            selected_dates = [self.data.index[i] for i in selected_indices]
        else:
            selected_dates = self.data.index
        
        # 重建收益率
        reconstructed = self.reconstruct_yields()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：实际收益率曲线
        for i, date in enumerate(selected_dates):
            actual_yields = self.original_data.loc[date]
            color = plt.cm.viridis(i / len(selected_dates))
            axes[0].plot(self.maturities_numeric, actual_yields.values,
                       color=color, alpha=0.7, linewidth=1.5,
                       label=date.strftime('%Y-%m'))
        
        axes[0].set_title('实际收益率曲线历史变化', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('期限（年）', fontsize=12)
        axes[0].set_ylabel('收益率', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 右图：PCA重建收益率曲线
        for i, date in enumerate(selected_dates):
            recon_yields = reconstructed.loc[date]
            color = plt.cm.viridis(i / len(selected_dates))
            axes[1].plot(self.maturities_numeric, recon_yields.values,
                       color=color, alpha=0.7, linewidth=1.5,
                       label=date.strftime('%Y-%m'))
        
        axes[1].set_title('PCA重建收益率曲线历史变化', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('期限（年）', fontsize=12)
        axes[1].set_ylabel('收益率', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 简化图例
        if len(selected_dates) > 6:
            for ax in axes:
                ax.legend(loc='upper right', ncol=2, fontsize=8)
        else:
            for ax in axes:
                ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_variance_explained(self):
        """绘制解释方差图"""
        explained_variance = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：各主成分解释方差
        bars = ax1.bar(range(1, len(explained_variance)+1), explained_variance,
                      color=plt.cm.viridis(np.linspace(0, 1, len(explained_variance))))
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('解释方差比例')
        ax1.set_title('各主成分解释方差')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值
        for i, v in enumerate(explained_variance):
            ax1.text(i+1, v+0.01, f'{v:.2%}', ha='center', va='bottom')
        
        # 右图：累计解释方差
        ax2.plot(range(1, len(cumulative_variance)+1), cumulative_variance,
                'b-o', linewidth=2, markersize=8)
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80%阈值')
        ax2.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90%阈值')
        ax2.set_xlabel('主成分数量')
        ax2.set_ylabel('累计解释方差比例')
        ax2.set_title('累计解释方差')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 在曲线上添加数值
        for i, v in enumerate(cumulative_variance):
            ax2.text(i+1, v+0.01, f'{v:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def run_comprehensive_analysis(self):
        """运行全面PCA分析"""
        print("="*60)
        print("中国国债PCA模型全面分析")
        print("="*60)
        
        # 1. 数据检查
        print("\n1. 数据检查...")
        print(f"数据时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        print(f"数据点数量: {len(self.data)}")
        print(f"可用期限: {list(self.data.columns)}")
        
        # 2. 拟合PCA模型
        print("\n2. 拟合PCA模型...")
        self.fit_pca_model()
        
        # 3. 绘制解释方差图
        print("\n3. 方差解释分析...")
        self.plot_variance_explained()
        
        # 4. 绘制因子时间序列
        print("\n4. 因子时间序列分析...")
        self.plot_factors_timeseries()
        
        # 5. 拟合精度评估
        print("\n5. 拟合精度评估...")
        print("最新日期拟合:")
        self.analyze_fitting_accuracy()
        
        # 6. 整体拟合质量
        print("\n6. 整体拟合质量...")
        metrics = self.calculate_overall_fitting_metrics()
        
        # 7. 历史收益率曲线对比
        print("\n7. 历史收益率曲线分析...")
        self.plot_historical_yield_curves_pca()
        
        # 8. 交易策略分析
        print("\n8. PCA交易策略分析...")
        trade_details = self.relative_value_analysis_pca(target_maturity='10Y')
        
        # 9. 模型比较
        print("\n9. 模型比较摘要...")
        print(f"PCA模型: {self.n_components}个主成分")
        print(f"累计解释方差: {np.sum(self.pca_model.explained_variance_ratio_):.2%}")
        print(f"平均拟合RMSE: {metrics['avg_rmse']:.4%} ({metrics['avg_rmse']*10000:.1f} bp)")
        
        return {
            'factors': self.factors_df,
            'loadings': self.loadings,
            'explained_variance': self.pca_model.explained_variance_ratio_,
            'metrics': metrics,
            'trade_details': trade_details
        }
    
    def save_results(self, output_path='pca_china_results.xlsx'):
        """保存PCA分析结果"""
        print(f"\n保存PCA分析结果到: {output_path}")
        
        # 1. 保存因子数据
        factors_df = self.factors_df.copy()
        factors_df['Date'] = factors_df.index
        factors_df = factors_df[['Date'] + list(factors_df.columns[:-1])]
        
        # 2. 保存载荷矩阵
        loadings_df = pd.DataFrame(
            self.loadings,
            index=[f'PC{i+1}' for i in range(self.n_components)],
            columns=self.data.columns
        )
        
        # 3. 保存解释方差
        variance_df = pd.DataFrame({
            '主成分': [f'PC{i+1}' for i in range(self.n_components)],
            '解释方差': self.pca_model.explained_variance_ratio_,
            '累计解释方差': np.cumsum(self.pca_model.explained_variance_ratio_)
        })
        
        # 4. 保存重建收益率
        reconstructed = self.reconstruct_yields()
        reconstructed_df = reconstructed.copy()
        reconstructed_df['Date'] = reconstructed_df.index
        reconstructed_df = reconstructed_df[['Date'] + list(reconstructed_df.columns[:-1])]
        
        # 5. 保存到Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            factors_df.to_excel(writer, sheet_name='PCA因子', index=False)
            loadings_df.to_excel(writer, sheet_name='因子载荷', index=True)
            variance_df.to_excel(writer, sheet_name='解释方差', index=False)
            reconstructed_df.to_excel(writer, sheet_name='重建收益率', index=False)
        
        print("已保存: PCA因子, 因子载荷, 解释方差, 重建收益率")
        return output_path


# 主程序
def main():
    """主程序 - PCA模型分析"""
    print("="*60)
    print("中国国债利率期限结构分析 - PCA模型")
    print("="*60)
    
    # 请将此处替换为您的数据文件路径
    data_file = 'CN_Yield_2018_2025.xlsx'
    
    try:
        # 创建PCA分析器实例
        analyzer = ChinaTreasuryPCAAnalysis(
            data_path=data_file,
            n_components=3  # 通常使用3个主成分
        )
        
        # 运行全面分析
        results = analyzer.run_comprehensive_analysis()
        
        # 保存结果
        output_file = analyzer.save_results('pca_china_results_detailed.xlsx')
        
        print("\n" + "="*60)
        print(f"PCA分析完成! 结果已保存到: {output_file}")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()



# 可选：比较PCA和Gauss+模型
def compare_models(pca_results, gaussplus_results):
    """比较PCA和Gauss+模型结果"""
    print("\n" + "="*60)
    print("模型比较: PCA vs Gauss+")
    print("="*60)
    
    if 'metrics' in pca_results and 'avg_rmse' in pca_results['metrics']:
        pca_rmse = pca_results['metrics']['avg_rmse']
        print(f"PCA模型平均RMSE: {pca_rmse:.4%} ({pca_rmse*10000:.1f} bp)")
    
    # 如果有Gauss+结果，可以进行比较
    # print(f"Gauss+模型平均RMSE: {gaussplus_rmse:.4%} ({gaussplus_rmse*10000:.1f} bp)")
    
    # 比较解释能力
    print(f"PCA累计解释方差: {np.sum(pca_results['explained_variance']):.2%}")
    
    # 比较交易信号
    if 'trade_details' in pca_results:
        pca_trades = len(pca_results['trade_details']) if pca_results['trade_details'] else 0
        print(f"PCA交易信号数量: {pca_trades}")


if __name__ == "__main__":
    main()