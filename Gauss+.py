import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, Bounds
from GaussPlusModel import GaussPlusModel
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ChinaTreasuryGaussPlusAnalysis:
    """
    使用Gauss+模型分析中国国债利率期限结构
    """
    
    def __init__(self, data_path, short_rate_col='1M', benchmark_cols=['2Y', '10Y']):
        """
        初始化
        
        Parameters
        ----------
        data_path : str
            数据文件路径
        short_rate_col : str
            短期利率列名（如1个月期）
        benchmark_cols : list
            基准收益率列名（2年和10年）
        """
        self.data_path = data_path
        self.short_rate_col = short_rate_col
        self.benchmark_cols = benchmark_cols
        
        # 读取数据
        self.load_data()
        
        # 模型参数初始化
        self.model = None
        self.factors = None
        
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
            # 检查数据范围，如果大部分值在0-10之间，很可能是百分比
            col_mean = self.data[col].mean()
            print(f"{col}: 均值 = {col_mean:.4f}")
            
            if col_mean > 1:  # 如果均值大于1，假设是百分比形式
                print(f"  → 将{col}从百分比转换为小数（除以100）")
                self.data[col] = self.data[col] / 100
        
        # 检查缺失值
        missing_pct = self.data.isnull().sum() / len(self.data) * 100
        missing_cols = missing_pct[missing_pct > 0]
        if len(missing_cols) > 0:
            print("\n缺失值统计:")
            print(missing_cols)
            # 使用前向填充处理缺失值
            self.data = self.data.ffill()
            print("已使用前向填充处理缺失值")
    
    def prepare_yield_changes(self, maturities):
        """
        准备收益率变化数据
        
        Parameters
        ----------
        maturities : list
            期限列表（以年为单位）
        """
        # 确保基准期限在数据中
        for col in self.benchmark_cols:
            if col not in self.data.columns:
                raise ValueError(f"基准期限 {col} 不在数据中")
        
        # 创建收益率变化矩阵
        delta_y = self.data[maturities].diff().dropna()
        delta_yb = self.data[self.benchmark_cols].diff().dropna()
        
        print(f"收益率变化数据维度: delta_y={delta_y.shape}, delta_yb={delta_yb.shape}")
        
        return delta_y.values, delta_yb.values
    
    def initialize_model(self):
        """初始化Gauss+模型参数"""
        # 基于中国国债市场的经验参数（调整后）
        # alpha: (alpha_r, alpha_m, alpha_l) - 均值回归速度
        # 中国政策利率调整相对平稳，短期均值回归较快
        alpha_init = [0.5, 0.2, 0.05]  # 调整为更有差异的值
        
        # sigma: (sigma_l, sigma_m, rho) - 波动率和相关性
        # 调整到更合理的水平
        sigma_init = [0.01, 0.015, 0.3]
        
        # mu: 长期均值 - 使用10年期收益率的长期均值
        mu_init = self.data[self.benchmark_cols[1]].mean()
        
        print(f"\n初始化模型参数:")
        print(f"alpha: {alpha_init}")
        print(f"sigma: {sigma_init}")
        print(f"mu: {mu_init:.4%}")
        
        self.model = GaussPlusModel(alpha_init, sigma_init, mu_init)
        
        return self.model
    
    def calibrate_model_improved(self, maturities=None):
        """
        改进的模型校准方法
        """
        if maturities is None:
            maturities = list(self.data.columns)
        
        print(f"\n开始改进的模型校准...")
        print(f"使用期限: {maturities}")
        
        # 获取数值型期限
        maturities_numeric = self.get_maturity_numeric(maturities)
        print(f"数值期限: {maturities_numeric}")
        
        # 准备数据
        delta_y, delta_yb = self.prepare_yield_changes(maturities)
        
        # 阶段1: 更稳健的alpha校准
        print("\n阶段1: 校准alpha参数...")
        
        def objective_alpha(alpha_vec):
            # 确保alpha为正且满足条件
            if any(alpha_vec <= 0) or any(alpha_vec > 10):
                return 1e10
            
            # 确保alpha值有足够差异
            if min(np.diff(sorted(alpha_vec))) < 0.05:
                return 1e10
            
            self.model.alpha = alpha_vec
            self.model.Ainv = self.model._A_inv()
            
            # 构建负载矩阵
            Ups_all = []
            Ups_b = []
            
            for tau in maturities_numeric:
                U = self.model.upsilon(tau)
                Ups_all.append(U[1:])
            
            for tau in [2.0, 10.0]:
                U = self.model.upsilon(tau)
                Ups_b.append(U[1:])
            
            Ups_all = np.asarray(Ups_all)
            Ups_b = np.asarray(Ups_b)
            
            if np.linalg.matrix_rank(Ups_b) < 2:
                return 1e10
            
            Ups_b_inv = np.linalg.inv(Ups_b)
            model_slopes = Ups_all @ Ups_b_inv
            
            # 计算OLS beta
            beta_hat = np.linalg.inv(delta_yb.T @ delta_yb) @ delta_yb.T @ delta_y
            
            return np.linalg.norm(model_slopes.T - beta_hat, 'fro')
        
        # 使用更强大的优化算法
        bounds = [(0.01, 2.0), (0.01, 1.0), (0.001, 0.5)]
        result = differential_evolution(
            objective_alpha,
            bounds,
            maxiter=100,
            popsize=20,
            seed=42
        )
        
        self.model.alpha = result.x
        self.model.Ainv = self.model._A_inv()
        
        print(f"校准后的alpha: {self.model.alpha}")
        print(f"目标函数值: {result.fun:.6f}")
        
        # 阶段2: sigma校准
        print("\n阶段2: 校准sigma参数...")
        
        def objective_sigma(sigma_vec):
            self.model.sigma = sigma_vec
            
            # 计算样本方差
            sample_var = np.cov(delta_yb.T) * 252  # 年化
            
            # 计算模型隐含方差
            Om = self.model._Omega()
            ups_b = np.vstack([self.model.upsilon(tau) for tau in [2.0, 10.0]])
            model_var = ups_b @ Om @ Om.T @ ups_b.T
            
            # 匹配方差
            return np.linalg.norm(model_var - sample_var, 'fro')
        
        sigma_bounds = [(0.001, 0.05), (0.001, 0.05), (-0.99, 0.99)]
        sigma_result = differential_evolution(
            objective_sigma,
            sigma_bounds,
            maxiter=50,
            popsize=15,
            seed=42
        )
        
        self.model.sigma = sigma_result.x
        self.model.Sigma = self.model._Sigma()
        print(f"校准后的sigma: {self.model.sigma}")
        
        # 阶段3: 提取因子并校准mu
        print("\n阶段3: 提取因子并校准mu...")
        self.extract_factors()
        
        # 校准mu
        print("\n阶段4: 校准mu参数...")
        mu_result = self.calibrate_mu_improved(maturities)
        
        print(f"\n最终模型参数:")
        print(f"alpha: {self.model.alpha}")
        print(f"sigma: {self.model.sigma}")
        print(f"mu: {self.model.mu:.4%}")
        
        return {
            'alpha': self.model.alpha,
            'sigma': self.model.sigma,
            'mu': self.model.mu,
            'factors': self.factors,
            'factor_dates': self.factor_dates
        }
    
    def extract_factors(self):
        """提取所有时间点的因子"""
        factors_list = []
        valid_dates = []
        
        print("提取因子...")
        
        for date in self.data.index:
            try:
                r_t = self.data.loc[date, self.short_rate_col]
                y2 = self.data.loc[date, self.benchmark_cols[0]]
                y10 = self.data.loc[date, self.benchmark_cols[1]]
                
                if not (np.isnan(r_t) or np.isnan(y2) or np.isnan(y10)):
                    factors = self.model.extract_factors(r_t, y2, y10)
                    factors_list.append(factors)
                    valid_dates.append(date)
            except Exception as e:
                continue
        
        self.factors = np.array(factors_list)
        self.factor_dates = valid_dates
        
        print(f"成功提取 {len(self.factors)} 个时间点的因子")
    
    def calibrate_mu_improved(self, maturities):
        """改进的mu校准"""
        # 获取数值型期限
        maturities_numeric = self.get_maturity_numeric(maturities)
        
        # 准备数据
        Y = self.data.loc[self.factor_dates, maturities].values
        X = self.factors
        
        def objective(mu_val):
            self.model.mu = mu_val[0]
            
            total_error = 0
            for i, x in enumerate(X):
                model_yields = np.array([self.model.zero_yield(tau, x) 
                                        for tau in maturities_numeric])
                actual_yields = Y[i]
                errors = actual_yields - model_yields
                total_error += np.sum(errors**2)
            
            return total_error / len(X)
        
        # 使用更合理的mu范围
        mu_min = np.percentile(self.data.values.flatten(), 10)
        mu_max = np.percentile(self.data.values.flatten(), 90)
        
        bounds = [(mu_min, mu_max)]
        result = differential_evolution(
            objective,
            bounds,
            maxiter=30,
            seed=42
        )
        
        self.model.mu = result.x[0]
        return self.model.mu
    
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
    
    def analyze_factors(self):
        """分析提取的因子"""
        if self.factors is None:
            print("请先校准模型以提取因子")
            return
        
        print("\n" + "="*60)
        print("因子分析结果")
        print("="*60)
        
        print(f"\n因子统计:")
        factor_names = ['短期因子(r_t)', '中期因子(m_t)', '长期因子(l_t)']
        
        stats_df = pd.DataFrame({
            '因子': factor_names,
            '均值': [f"{self.factors[:, i].mean():.4%}" for i in range(3)],
            '标准差': [f"{self.factors[:, i].std():.4%}" for i in range(3)],
            '最小值': [f"{self.factors[:, i].min():.4%}" for i in range(3)],
            '最大值': [f"{self.factors[:, i].max():.4%}" for i in range(3)]
        })
        
        print(stats_df.to_string(index=False))
        
        # 因子相关性
        factor_corr = np.corrcoef(self.factors.T)
        print(f"\n因子相关性矩阵:")
        print("            r_t       m_t       l_t")
        for i, (label, row) in enumerate(zip(['r_t', 'm_t', 'l_t'], factor_corr)):
            print(f"{label:>4}: [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
    
    def plot_factors_with_components(self):
        """绘制因子时间序列及其分解"""
        if self.factors is None:
            print("请先校准模型以提取因子")
            return
        
        # 计算因子的滚动统计
        factors_df = pd.DataFrame(self.factors, 
                                 index=self.factor_dates,
                                 columns=['短期因子(r_t)', '中期因子(m_t)', '长期因子(l_t)'])
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # 1. 短期因子
        axes[0].plot(factors_df.index, factors_df['短期因子(r_t)'], 'b-', linewidth=1.5)
        axes[0].axhline(y=factors_df['短期因子(r_t)'].mean(), color='r', linestyle='--', alpha=0.5)
        axes[0].set_title('短期因子 (r_t) - 政策利率因子', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('利率')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(['短期因子', f"均值={factors_df['短期因子(r_t)'].mean():.2%}"], loc='upper right')
        
        # 2. 中期因子
        axes[1].plot(factors_df.index, factors_df['中期因子(m_t)'], 'g-', linewidth=1.5)
        axes[1].axhline(y=factors_df['中期因子(m_t)'].mean(), color='r', linestyle='--', alpha=0.5)
        axes[1].set_title('中期因子 (m_t) - 货币政策预期因子', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('利率')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(['中期因子', f"均值={factors_df['中期因子(m_t)'].mean():.2%}"], loc='upper right')
        
        # 3. 长期因子
        axes[2].plot(factors_df.index, factors_df['长期因子(l_t)'], 'r-', linewidth=1.5)
        axes[2].axhline(y=factors_df['长期因子(l_t)'].mean(), color='r', linestyle='--', alpha=0.5)
        axes[2].axhline(y=self.model.mu, color='orange', linestyle='-', alpha=0.7, linewidth=2)
        axes[2].set_title('长期因子 (l_t) - 长期预期因子', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('利率')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(['长期因子', f"均值={factors_df['长期因子(l_t)'].mean():.2%}", 
                       f"模型μ={self.model.mu:.2%}"], loc='upper right')
        
        # 4. 所有因子一起
        axes[3].plot(factors_df.index, factors_df['短期因子(r_t)'], 'b-', alpha=0.7, label='短期')
        axes[3].plot(factors_df.index, factors_df['中期因子(m_t)'], 'g-', alpha=0.7, label='中期')
        axes[3].plot(factors_df.index, factors_df['长期因子(l_t)'], 'r-', alpha=0.7, label='长期')
        axes[3].set_title('三因子对比', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('日期')
        axes[3].set_ylabel('利率')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # 额外：绘制因子分布的直方图
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, (ax, col) in enumerate(zip(axes, factors_df.columns)):
            ax.hist(factors_df[col], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=factors_df[col].mean(), color='red', linestyle='--', linewidth=2)
            ax.set_title(f'{col}分布')
            ax.set_xlabel('利率')
            ax.set_ylabel('频数')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_yield_curve_fit_detailed(self, date=None):
        """
        绘制详细的收益率曲线拟合情况
        """
        if date is None:
            date_idx = -1
            date = self.factor_dates[-1]
        else:
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            date_idx = np.where([d.strftime('%Y-%m-%d') for d in self.factor_dates] == date_str)[0]
            if len(date_idx) == 0:
                print(f"日期 {date} 不在因子数据中")
                return
            date_idx = date_idx[0]
        
        x = self.factors[date_idx]
        actual_date = self.factor_dates[date_idx]
        
        # 获取实际收益率
        actual_yields = self.data.loc[actual_date]
        
        # 计算模型预测
        actual_maturities_numeric = self.get_maturity_numeric(actual_yields.index.tolist())
        predicted_yields = [self.model.zero_yield(tau, x) for tau in actual_maturities_numeric]
        
        # 生成平滑的模型曲线
        smooth_maturities = np.linspace(0.08, 30, 200)  # 从1个月到30年
        smooth_model_yields = [self.model.zero_yield(tau, x) for tau in smooth_maturities]
        
        # 计算拟合误差
        errors = np.array(actual_yields.values) - np.array(predicted_yields)
        rmse = np.sqrt(np.mean(errors**2))
        
        # 绘制详细图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：收益率曲线拟合
        ax1.scatter(actual_maturities_numeric, actual_yields.values, 
                   color='red', s=80, zorder=5, label='实际收益率', alpha=0.8)
        ax1.plot(smooth_maturities, smooth_model_yields, 'b-', 
                linewidth=2.5, label='Gauss+模型拟合', alpha=0.8)
        
        # 连接实际点与预测点，显示误差
        for tau_act, y_act, y_pred in zip(actual_maturities_numeric, 
                                         actual_yields.values, predicted_yields):
            ax1.plot([tau_act, tau_act], [y_act, y_pred], 'gray', linestyle='--', alpha=0.5)
        
        ax1.set_title(f'收益率曲线拟合 - {actual_date.strftime("%Y-%m-%d")}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('期限（年）', fontsize=12)
        ax1.set_ylabel('收益率', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 标注关键点
        for i, (tau, y_act, y_pred) in enumerate(zip(actual_maturities_numeric, 
                                                    actual_yields.values, predicted_yields)):
            ax1.annotate(f'{y_act:.2%}', (tau, y_act), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=9, color='red')
            ax1.annotate(f'{y_pred:.2%}', (tau, y_pred), 
                        textcoords="offset points", xytext=(0,-15), 
                        ha='center', fontsize=9, color='blue')
        
        # 右图：拟合误差
        bars = ax2.bar(range(len(errors)), errors*10000, 
                      color=['red' if e>0 else 'blue' for e in errors], alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax2.set_title(f'拟合误差分析 (RMSE={rmse:.4%})', fontsize=14, fontweight='bold')
        ax2.set_xlabel('期限点')
        ax2.set_ylabel('误差（基点）')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加误差值标注
        for i, (bar, error) in enumerate(zip(bars, errors)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error*10000:.1f}bp', ha='center', va='bottom' if height>0 else 'top',
                    fontsize=9)
        
        # 设置x轴刻度
        ax2.set_xticks(range(len(errors)))
        ax2.set_xticklabels(actual_yields.index, rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细统计
        print(f"\n详细拟合误差分析 ({actual_date.strftime('%Y-%m-%d')}):")
        print("="*60)
        print(f"{'期限':<8} {'实际值':<10} {'预测值':<10} {'误差(bp)':<12} {'相对误差':<12}")
        print("-"*60)
        
        for tau, y_act, y_pred, error in zip(actual_yields.index, 
                                           actual_yields.values, 
                                           predicted_yields, 
                                           errors):
            rel_error = abs(error/y_act) if y_act != 0 else np.nan
            print(f"{tau:<8} {y_act:>8.4%} {y_pred:>8.4%} {error*10000:>10.1f} {rel_error:>11.2%}")
        
        print("-"*60)
        print(f"RMSE: {rmse:.4%} ({rmse*10000:.1f} bp)")
        print(f"最大绝对误差: {np.max(np.abs(errors)):.4%} ({np.max(np.abs(errors))*10000:.1f} bp)")
        print(f"平均绝对误差: {np.mean(np.abs(errors)):.4%} ({np.mean(np.abs(errors))*10000:.1f} bp)")
    
    def relative_value_analysis_improved(self, forward_term=9, window_fast=5, window_slow=40):
        """
        改进的相对价值交易分析
        """
        if self.factors is None:
            print("请先校准模型以提取因子")
            return
        
        print(f"\n改进的相对价值分析: {forward_term}年期远期利率")
        
        # 计算每个时间点的拟合误差（使用所有期限）
        fitting_errors_list = []
        
        for i, (date, x) in enumerate(zip(self.factor_dates, self.factors)):
            # 获取该日期的所有实际收益率
            actual_yields = self.data.loc[date]
            actual_maturities = self.get_maturity_numeric(actual_yields.index.tolist())
            
            # 计算模型预测
            predicted_yields = [self.model.zero_yield(tau, x) for tau in actual_maturities]
            
            # 计算整体拟合误差（所有期限的RMSE）
            errors = np.array(actual_yields.values) - np.array(predicted_yields)
            overall_error = np.sqrt(np.mean(errors**2))
            
            # 特别计算远期利率的拟合误差
            # 对于远期利率，我们需要市场实际远期利率数据
            # 这里使用简化的代理：长期利率的拟合误差
            if '10Y' in actual_yields.index:
                forward_error = errors[list(actual_yields.index).index('10Y')]
            else:
                forward_error = errors[-1]  # 使用最后一个期限
            
            fitting_errors_list.append(forward_error)
        
        fitting_errors = np.array(fitting_errors_list)
        
        if len(fitting_errors) < window_slow * 2:
            print(f"数据点不足，需要至少 {window_slow*2} 个数据点，当前只有 {len(fitting_errors)} 个")
            return
        
        # 计算移动平均信号
        ma_fast = pd.Series(fitting_errors).rolling(window=window_fast, min_periods=1).mean()
        ma_slow = pd.Series(fitting_errors).rolling(window=window_slow, min_periods=1).mean()
        signal = ma_fast - ma_slow
        
        # 去中心化的拟合误差
        demeaned_errors = fitting_errors - np.mean(fitting_errors)
        
        # 设置交易阈值
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
                    trade_signals.append(1)  # 买入信号（实际利率高于模型预测）
                    trade_type = "买入"
                elif demeaned_errors[i] < 0 and signal[i] > signal_threshold:
                    trade_signals.append(-1)  # 卖出信号（实际利率低于模型预测）
                    trade_type = "卖出"
                else:
                    trade_signals.append(0)
                    trade_type = None
                
                if trade_type:
                    trade_dates.append(self.factor_dates[i])
                    trade_details.append({
                        'date': self.factor_dates[i],
                        'type': trade_type,
                        'error': demeaned_errors[i],
                        'signal': signal[i],
                        'actual_rate': self.data.loc[self.factor_dates[i], '10Y'] if '10Y' in self.data.columns else np.nan
                    })
            else:
                trade_signals.append(0)
        
        # 绘制结果
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. 拟合误差时间序列
        axes[0].plot(self.factor_dates, demeaned_errors*10000, 'b-', linewidth=1.5, alpha=0.7)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].axhline(y=error_threshold*10000, color='g', linestyle=':', alpha=0.5)
        axes[0].axhline(y=-error_threshold*10000, color='g', linestyle=':', alpha=0.5)
        axes[0].fill_between(self.factor_dates, 
                           -error_threshold*10000, 
                           error_threshold*10000, 
                           alpha=0.1, color='gray')
        axes[0].set_title(f'去中心化拟合误差 (阈值=±{error_threshold*10000:.1f}bp)', 
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('误差（基点）')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(['拟合误差', '零线', f'阈值±{error_threshold*10000:.1f}bp'])
        
        # 2. 交易信号
        axes[1].plot(self.factor_dates, signal*10000, 'g-', linewidth=1.5, alpha=0.7, label='信号')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=signal_threshold*10000, color='orange', linestyle=':', alpha=0.5)
        axes[1].axhline(y=-signal_threshold*10000, color='orange', linestyle=':', alpha=0.5)
        axes[1].fill_between(self.factor_dates, 
                           -signal_threshold*10000, 
                           signal_threshold*10000, 
                           alpha=0.1, color='gray')
        axes[1].set_title(f'交易信号 (阈值=±{signal_threshold*10000:.1f}bp)', 
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('信号强度（基点）')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(['信号', '零线', f'阈值±{signal_threshold*10000:.1f}bp'])
        
        # 3. 交易信号点
        axes[2].plot(self.factor_dates, np.zeros_like(demeaned_errors), 'k-', alpha=0.3)
        
        # 标记交易点
        buy_dates = [td['date'] for td in trade_details if td['type'] == '买入']
        sell_dates = [td['date'] for td in trade_details if td['type'] == '卖出']
        
        if buy_dates:
            buy_errors = [td['error']*10000 for td in trade_details if td['type'] == '买入']
            axes[2].scatter(buy_dates, [0]*len(buy_dates), 
                          color='green', s=100, marker='^', label='买入信号', zorder=5)
        
        if sell_dates:
            sell_errors = [td['error']*10000 for td in trade_details if td['type'] == '卖出']
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
        print(f"\n交易信号统计:")
        print("="*60)
        print(f"总交易日数: {len(self.factor_dates)}")
        print(f"交易信号数: {len(trade_details)}")
        print(f"买入信号: {len(buy_dates)}")
        print(f"卖出信号: {len(sell_dates)}")
        print(f"信号频率: {len(trade_details)/len(self.factor_dates):.2%}")
        
        if trade_details:
            print(f"\n最新交易信号:")
            for i, trade in enumerate(trade_details[-5:]):  # 显示最近5个信号
                print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['type']} "
                      f"(误差={trade['error']*10000:.1f}bp, "
                      f"信号={trade['signal']*10000:.1f}bp)")
        
        # 计算简单的策略回测（假设每次交易持有一个月）
        if len(trade_details) >= 2:
            returns = []
            for i in range(len(trade_details)-1):
                trade = trade_details[i]
                next_trade = trade_details[i+1]
                
                if trade['type'] == '买入':
                    # 买入策略：期望误差缩小
                    if abs(next_trade['error']) < abs(trade['error']):
                        returns.append(1)  # 成功
                    else:
                        returns.append(-1)  # 失败
                elif trade['type'] == '卖出':
                    # 卖出策略：期望误差扩大（符号变化）
                    if abs(next_trade['error']) > abs(trade['error']):
                        returns.append(1)
                    else:
                        returns.append(-1)
            
            if returns:
                win_rate = sum([1 for r in returns if r > 0]) / len(returns)
                print(f"\n简单回测结果 (假设每次交易持有到下一个信号):")
                print(f"  总交易次数: {len(returns)}")
                print(f"  胜率: {win_rate:.2%}")
                print(f"  平均持有期: {len(self.factor_dates)/len(trade_details):.1f} 天")
        
        return trade_details
    
    def run_comprehensive_analysis(self):
        """运行全面分析"""
        print("="*60)
        print("中国国债Gauss+模型全面分析")
        print("="*60)
        
        # 1. 数据检查
        print("\n1. 数据检查...")
        print(f"数据时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        print(f"数据点数量: {len(self.data)}")
        print(f"可用期限: {list(self.data.columns)}")
        
        # 2. 初始化模型
        print("\n2. 初始化模型...")
        self.initialize_model()
        
        # 3. 校准模型
        print("\n3. 校准模型参数...")
        results = self.calibrate_model_improved()
        
        # 4. 因子分析
        print("\n4. 因子分析...")
        self.analyze_factors()
        
        # 5. 可视化
        print("\n5. 生成可视化图表...")
        self.plot_factors_with_components()
        
        # 6. 拟合效果评估
        print("\n6. 拟合效果评估...")
        print("最新日期拟合:")
        self.plot_yield_curve_fit_detailed()
        
        # 评估多个日期的平均拟合误差
        print("\n整体拟合质量评估...")
        all_errors = []
        for i, (date, x) in enumerate(zip(self.factor_dates, self.factors)):
            actual_yields = self.data.loc[date]
            actual_maturities = self.get_maturity_numeric(actual_yields.index.tolist())
            predicted_yields = [self.model.zero_yield(tau, x) for tau in actual_maturities]
            errors = np.array(actual_yields.values) - np.array(predicted_yields)
            all_errors.append(np.sqrt(np.mean(errors**2)))
        
        avg_rmse = np.mean(all_errors)
        print(f"平均RMSE: {avg_rmse:.4%} ({avg_rmse*10000:.1f} bp)")
        print(f"RMSE范围: {np.min(all_errors)*10000:.1f} - {np.max(all_errors)*10000:.1f} bp")
        
        # 7. 交易策略分析
        print("\n7. 相对价值交易策略分析...")
        trade_details = self.relative_value_analysis_improved()
        
        # 8. 模型诊断
        print("\n8. 模型诊断...")
        self.model_diagnostics()
        
        return results
    
    def model_diagnostics(self):
        """模型诊断"""
        print("\n" + "="*60)
        print("模型诊断")
        print("="*60)
        
        # 检查参数合理性
        print("\n1. 参数合理性检查:")
        
        # alpha检查
        alpha_ratio = self.model.alpha[0] / self.model.alpha[2]
        print(f"  alpha_r/alpha_l = {alpha_ratio:.2f} (建议>5)")
        
        # sigma检查
        print(f"  sigma_m/sigma_l = {self.model.sigma[1]/self.model.sigma[0]:.2f}")
        
        # mu检查
        long_rate_mean = self.factors[:, 2].mean()
        print(f"  mu与长期因子均值的差异: {abs(self.model.mu - long_rate_mean)*10000:.1f} bp")
        
        # 检查矩阵条件数
        Ainv_cond = np.linalg.cond(self.model.Ainv)
        Sigma_cond = np.linalg.cond(self.model.Sigma)
        print(f"\n2. 矩阵条件数:")
        print(f"  Ainv条件数: {Ainv_cond:.2e}")
        print(f"  Sigma条件数: {Sigma_cond:.2e}")
        
        # 检查因子稳定性
        print(f"\n3. 因子稳定性检查:")
        factor_vol = self.factors.std(axis=0) * np.sqrt(252)  # 年化波动率
        print(f"  因子年化波动率: {factor_vol}")
        
        # 检查自相关性
        from statsmodels.tsa.stattools import acf
        for i, name in enumerate(['r_t', 'm_t', 'l_t']):
            autocorr = acf(self.factors[:, i], nlags=5)[1:]  # 去掉lag=0
            significant_lags = np.where(np.abs(autocorr) > 2/np.sqrt(len(self.factors)))[0]
            if len(significant_lags) > 0:
                print(f"  {name}在滞后{significant_lags+1}处有显著自相关")
        
        print("\n诊断完成!")

    def save_results_improved(self, output_path='gaussplus_china_results_detailed.xlsx'):
        """保存详细分析结果到Excel文件"""
        if self.factors is None:
            print("请先校准模型")
            return
        
        print(f"\n保存详细结果到: {output_path}")
        
        # 1. 保存因子数据
        factors_df = pd.DataFrame({
            'Date': self.factor_dates,
            'Short_Factor': self.factors[:, 0],
            'Medium_Factor': self.factors[:, 1],
            'Long_Factor': self.factors[:, 2]
        })
        
        # 2. 保存模型参数
        params_df = pd.DataFrame({
            'Parameter': ['alpha_r', 'alpha_m', 'alpha_l', 
                        'sigma_l', 'sigma_m', 'rho', 'mu'],
            'Value': [self.model.alpha[0], self.model.alpha[1], self.model.alpha[2],
                    self.model.sigma[0], self.model.sigma[1], self.model.sigma[2],
                    self.model.mu],
            'Description': [
                '短期因子均值回归速度',
                '中期因子均值回归速度', 
                '长期因子均值回归速度',
                '长期因子波动率',
                '中期因子波动率',
                '中期-长期因子相关性',
                '长期均值水平'
            ]
        })
        
        # 3. 保存拟合误差统计
        error_stats = []
        dates_to_evaluate = self.factor_dates[-10:]  # 最近10个交易日
        
        for date in dates_to_evaluate:
            date_str = date.strftime('%Y-%m-%d')
            if date in self.factor_dates:
                idx = list(self.factor_dates).index(date)
                x = self.factors[idx]
                actual_yields = self.data.loc[date]
                maturities_numeric = self.get_maturity_numeric(actual_yields.index.tolist())
                
                # 计算各期限误差
                for maturity, tau in zip(actual_yields.index, maturities_numeric):
                    y_actual = actual_yields[maturity]
                    y_pred = self.model.zero_yield(tau, x)
                    error_bp = (y_actual - y_pred) * 10000
                    
                    error_stats.append({
                        'Date': date_str,
                        'Maturity': maturity,
                        'Actual_Yield': y_actual,
                        'Predicted_Yield': y_pred,
                        'Error_bp': error_bp,
                        'Abs_Error_bp': abs(error_bp)
                    })
        
        error_df = pd.DataFrame(error_stats)
        
        # 4. 汇总统计
        summary_stats = pd.DataFrame({
            'Metric': ['平均RMSE(bp)', '最大绝对误差(bp)', '平均绝对误差(bp)', 
                    '短期拟合误差(bp)', '长期拟合误差(bp)', '数据点数'],
            'Value': [np.mean(self.calculate_overall_rmse()) * 10000,
                    np.max([abs(e) for e in self.calculate_all_errors()]) * 10000,
                    np.mean([abs(e) for e in self.calculate_all_errors()]) * 10000,
                    self.calculate_short_term_error(),
                    self.calculate_long_term_error(),
                    len(self.factors)]
        })
        
        # 5. 保存到Excel的不同sheet
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            factors_df.to_excel(writer, sheet_name='因子序列', index=False)
            params_df.to_excel(writer, sheet_name='模型参数', index=False)
            error_df.to_excel(writer, sheet_name='拟合误差详情', index=False)
            summary_stats.to_excel(writer, sheet_name='汇总统计', index=False)
            
            # 如果有交易信号，也保存
            try:
                trade_details = self.relative_value_analysis_improved()
                if trade_details:
                    trade_df = pd.DataFrame(trade_details)
                    trade_df.to_excel(writer, sheet_name='交易信号', index=False)
            except:
                pass
        
        print(f"已保存: 因子序列({len(factors_df)}行), 模型参数, 拟合误差详情({len(error_df)}行)")

    def calculate_overall_rmse(self):
        """计算整体RMSE"""
        all_errors = []
        for i, (date, x) in enumerate(zip(self.factor_dates, self.factors)):
            actual_yields = self.data.loc[date]
            actual_maturities = self.get_maturity_numeric(actual_yields.index.tolist())
            predicted_yields = [self.model.zero_yield(tau, x) for tau in actual_maturities]
            errors = np.array(actual_yields.values) - np.array(predicted_yields)
            rmse = np.sqrt(np.mean(errors**2))
            all_errors.append(rmse)
        return all_errors

    def calculate_all_errors(self):
        """计算所有误差"""
        all_errors = []
        for i, (date, x) in enumerate(zip(self.factor_dates, self.factors)):
            actual_yields = self.data.loc[date]
            actual_maturities = self.get_maturity_numeric(actual_yields.index.tolist())
            predicted_yields = [self.model.zero_yield(tau, x) for tau in actual_maturities]
            errors = actual_yields.values - predicted_yields
            all_errors.extend(errors)
        return all_errors

    def calculate_short_term_error(self):
        """计算短期（1年以内）平均误差"""
        short_term_errors = []
        for i, (date, x) in enumerate(zip(self.factor_dates, self.factors)):
            actual_yields = self.data.loc[date]
            for col in ['1M', '3M', '6M', '1Y']:
                if col in actual_yields.index:
                    tau = self.get_maturity_numeric([col])[0]
                    y_actual = actual_yields[col]
                    y_pred = self.model.zero_yield(tau, x)
                    short_term_errors.append(abs(y_actual - y_pred))
        return np.mean(short_term_errors) * 10000 if short_term_errors else 0

    def calculate_long_term_error(self):
        """计算长期（10年以上）平均误差"""
        long_term_errors = []
        for i, (date, x) in enumerate(zip(self.factor_dates, self.factors)):
            actual_yields = self.data.loc[date]
            for col in ['10Y', '30Y']:
                if col in actual_yields.index:
                    tau = self.get_maturity_numeric([col])[0]
                    y_actual = actual_yields[col]
                    y_pred = self.model.zero_yield(tau, x)
                    long_term_errors.append(abs(y_actual - y_pred))
        return np.mean(long_term_errors) * 10000 if long_term_errors else 0
    

# 主程序
def main():
    # ============================================
    # 1. 数据准备和加载
    # ============================================
    print("="*60)
    print("中国国债利率期限结构分析 - Gauss+模型 (改进版)")
    print("="*60)
    
    # 请将此处替换为您的数据文件路径
    data_file = 'CN_Yield_2018_2025.xlsx'  # 您的数据文件
    
    try:
        # 创建分析器实例
        analyzer = ChinaTreasuryGaussPlusAnalysis(
            data_path=data_file,
            short_rate_col='1M',  # 短期利率使用1个月期
            benchmark_cols=['2Y', '10Y']  # 基准使用2年和10年
        )
        
        # 运行全面分析
        results = analyzer.run_comprehensive_analysis()
        
        # 保存结果
        analyzer.save_results_improved('gaussplus_china_results_detailed.xlsx')
        
        print("\n" + "="*60)
        print("分析完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()