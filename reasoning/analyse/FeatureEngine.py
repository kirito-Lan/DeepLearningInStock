# 股票特征工程
import asyncio
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from config.LoguruConfig import log
from constant.ExponentEnum import ExponentEnum
from constant.MaroDataEnum import MacroDataEnum
from manager import StockDataManage, MacroDataManage
from model.entity.BaseMeta.BaseMeta import database
from utils.ReFormatDate import format_date


# 股票数据的特征工程
async def feature_engineering(stock_code: str, start_date: str, end_date: str):
    # 为每个股票代码单独一个文件夹存放图片
    try:
        stock = ExponentEnum.get_enum_by_code(stock_code)
        if stock is None:
            raise ValueError(f"股票代码{stock_code}不存在")
        else:
            os.makedirs(f"../picture/{stock_code}", exist_ok=True)
            os.makedirs(f"../processed_data/{stock_code}", exist_ok=True)
    except Exception as e:
        log.error("本系统中不存在该股票", e)
    # 获取合并的数据
    merged_data = await get_merged_data(end_date, start_date, stock_code)

    # 1.时间序列分解
    print("Existing columns:", merged_data.columns.tolist())
    decompose_time_series(stock_data=merged_data)
    print("Existing columns:", merged_data.columns.tolist())
    # 2.股票的价格和回报率特征
    reward_rate_feature(stock_data=merged_data)
    print("Existing columns:", merged_data.columns.tolist())
    # 3.交易量的特征
    volume_abnormal_feature(stock_data=merged_data, stock_code=stock_code)
    print("Existing columns:", merged_data.columns.tolist())
    # 4.窗口函数计算长期和短期的均值和波动性
    stock_window_feature(merged_data=merged_data, stock_code=stock_code)
    print("Existing columns:", merged_data.columns.tolist())
    # 5.宏观数据窗口特征
    merged_data=macro_window_feature(merged_data=merged_data, stock_code=stock_code)
    print("Existing columns:", merged_data.columns.tolist())

    # 滞后和交叉特征
    lag_cross_feature(merged_data=merged_data)
    # 归一化
    normalize_data(merged_data=merged_data,stock_code=stock_code)

    # 保存数据
    merged_data.fillna(0, inplace=True)
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    merged_data.to_csv(f"../processed_data/{stock_code}/feature_{stock_code}-{start_date}-{end_date}.csv")


async def get_merged_data(end_date, start_date, stock_code):
    """
    获取合并并且处理后股票和宏观数据的数据
    """
    # 获取股票数据
    start_date, end_date = format_date(start_date=start_date, end_date=end_date)
    stock_data = await StockDataManage.get_stock_data_local(stock_code=stock_code, start_date=start_date, end_date=end_date)
    stock_data["trade_date"] = pd.to_datetime(stock_data['trade_date'], format='%Y-%m-%d', errors='coerce')
    stock_data = stock_data.set_index('trade_date')
    # 数据库获取的CPI数据是月率MOM  PPI是年率 YOY PMI是原始数据需要计算的话需要进行转换恢复原始值
    start = stock_data.index[0]
    end = stock_data.index[-1]
    log.info(f"获取到数据库最新的股票起止时间：Start: {start}, End: {end}")
    cpi_data = await MacroDataManage.get_macro_data_local(types=MacroDataEnum.CPI, start_date=start, end_date=end)
    cpi_data = cpi_data.set_index('report_date')
    ppi_data = await MacroDataManage.get_macro_data_local(types=MacroDataEnum.PPI, start_date=start, end_date=end)
    ppi_data = ppi_data.set_index('report_date')
    pmi_data = await MacroDataManage.get_macro_data_local(types=MacroDataEnum.PMI, start_date=start, end_date=end)
    pmi_data = pmi_data.set_index('report_date')

    # 计算原始cpi ppi数据
    def _compute_index(x):
        """
        将CPI和PPI比率指标还原成原始值
        默认上月=100计算 数据库存储的数据单位是  %
        Args:
            x: 当前值
        """
        # 根据公式：指数 = 100 + X
        result = Decimal('100') + Decimal(x)
        # 舍入到两位小数（half-up）
        return result.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)

    cpi_data['CPI'] = cpi_data['current_value'].apply(_compute_index)
    ppi_data['PPI'] = ppi_data['current_value'].apply(_compute_index)
    # 重置列名
    cpi_data.columns = ['CPI_MOM', 'CPI']
    ppi_data.columns = ['PPI_YOY', 'PPI']
    pmi_data.columns = ['PMI']
    stock_data.columns = ['Open', 'Close', 'High', 'Low', 'Volume']

    # 合并宏观数据和股票数据
    def _combined_macro_stock():
        """合并所有的宏观数据和股票数据
        同时填充宏观数据产生的Nan值
        """
        # 得到已对齐的宏观数据
        macro_data = (pd.merge(left=cpi_data, right=ppi_data, left_index=True, right_index=True, how='outer')
                      .merge(right=pmi_data, left_index=True, right_index=True, how='outer'))
        # 调试时可以查看合并结果
        print("Macro Data 合并结果：\n")
        print(macro_data)
        # 以股票数据为主表合并宏观数据
        merged = stock_data.join(macro_data, how='left')
        # 以股票数据为主合并宏观数据后，对缺失值处理：
        # 对 ['CPI','PPI','PMI'] 列采用前向填充
        for col in ['CPI', 'PPI', 'PMI']:
            merged[col] = merged[col].ffill().bfill()
        # 删除['CPI_MOM', 'PPI_YOY']
        merged.drop(columns=['CPI_MOM', 'PPI_YOY'], axis=1, inplace=True)
        # 找出插值和填充后还存在nan的值
        missing_values = merged.isnull().sum()
        print("Missing Values: \n{}".format(missing_values))
        return merged

    merged_data = _combined_macro_stock()
    print("merged_data:\n{}".format(merged_data))
    # 返回合并后的数据
    return merged_data


# 时间序列分解 seasonal_decompose和 波动计算
def decompose_time_series(stock_data: pd.DataFrame):
    """
    对时间序列进行分解，并绘制分解结果
    以252个股票交易日为基准，使用乘法模型计算股票的数据
    同时计算20天的滚动标准差作为波动性特征
    Args:
        stock_data: 待分解的时间序列数据
    Returns:
        None
    """
    # 对股票收盘价数据进行季节性分解
    decomposition = seasonal_decompose(stock_data['Close'], model='multiplicative', period=252)
    trend = decomposition.trend  # 趋势
    seasonality = decomposition.seasonal  # 季节性
    residual = decomposition.resid  # 残差
    stock_data['Close_Trend'] = trend
    stock_data['Close_Seasonal'] = seasonality
    stock_data['Close_Residual'] = residual
    # 计算波动性（20天滚动标准差）
    stock_data['Close_Volatility'] = stock_data['Close'].rolling(window=20).std()
    # 对CPI PPI 和 PMI 的季节性分解
    for column in ['CPI', 'PPI', 'PMI']:
        decomposition = seasonal_decompose(stock_data[column], model='multiplicative', period=365)
        trend = decomposition.trend
        seasonality = decomposition.seasonal
        residual = decomposition.resid
        stock_data[f'{column}_Trend'] = trend
        stock_data[f'{column}_Seasonal'] = seasonality
        stock_data[f'{column}_Residual'] = residual
    # 计算波动性（30天滚动标准差）
    for column in ['CPI', 'PPI', 'PMI']:
        stock_data[f'{column}_Volatility'] = stock_data[column].rolling(window=30).std()


# 股票的交易量特征和异常值检测修复
def volume_abnormal_feature(stock_data: pd.DataFrame, stock_code: str):
    """
    计算股票的交易量特征,交易量的变化率
    处理异常的交易数据 Z-Score方法标记
    同时使用四分位法寻找异常值，然后用线性插值修复
    Args:
        stock_data: 包含股票交易量数据的DataFrame
        stock_code: 股票代码
    """
    # 计算交易量的变化率
    stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
    # 计算交易量的Z-score
    stock_data['Volume_Zscore'] = stats.zscore(stock_data['Volume'])

    # 设定阈值，Z-score超过3或小于-3的交易量被视为异常交易量
    stock_data['Volume_Anomaly_Zscore'] = (stock_data['Volume_Zscore'].abs() > 3).astype(int)

    # 交易量 IQR 异常值检测
    Q1_volume = stock_data['Volume'].quantile(0.25)
    Q3_volume = stock_data['Volume'].quantile(0.75)
    IQR_volume = Q3_volume - Q1_volume

    lower_bound_volume = Q1_volume - 1.5 * IQR_volume
    upper_bound_volume = Q3_volume + 1.5 * IQR_volume

    # 检测交易量中的异常值 布尔索引会返回Series的布尔值
    stock_data['Volume_outlier'] = ((stock_data['Volume'] < lower_bound_volume) |
                                    (stock_data['Volume'] > upper_bound_volume))
    # 对异常值进行修复
    stock_data['Volume_corrected'] = stock_data['Volume']
    # 将Volume_corrected列中标记的异常值设为 NaN
    stock_data.loc[stock_data['Volume_outlier'], 'Volume_corrected'] = np.nan  # 将异常值设为 NaN
    stock_data['Volume_corrected'] = stock_data['Volume_corrected'].interpolate()  # 使用插值法修复

    # 画图对比修复前后的图
    # 子图1: 修复前的交易量
    plt.subplot(211)
    plt.plot(stock_data.index, stock_data['Volume'], label='Original Volume', color='blue')
    # stock_data['Volume'][stock_data['Volume_outlier']]  df['列'][布尔条件]
    plt.scatter(stock_data.index[stock_data['Volume_outlier']],
                stock_data.loc[stock_data['Volume_outlier'], 'Volume'],
                color='red', label='Detected Outliers')
    plt.title('Volume Before Correction')
    plt.legend()
    # 手动设置 y 轴范围，为 [0, 1e9]
    plt.ylim(0, 1e9)

    # 子图2：修复后的交易量
    plt.subplot(212)
    plt.plot(stock_data.index, stock_data['Volume_corrected'], label='Corrected Volume', color='green')
    plt.title('Volume After Correction')
    plt.legend()
    # 同样设置相同的 y 轴范围
    plt.ylim(0, 1e9)
    plt.tight_layout()
    plt.savefig(f'../picture/{stock_code}/volume_correction.svg')
    plt.show()  # 显示图像后再关闭图表
    plt.close()


# 股票的价格和回报率特征
def reward_rate_feature(stock_data: pd.DataFrame):
    """
    计算股票价格和回报率特征
    对每日收盘价计算对数收益率，提取每日涨跌幅
    Args:
        stock_data: 包含股票价格数据的DataFrame
    """
    # 如果 'Close' 是 Decimal 类型，先转换为 float
    stock_data['Close_Float'] = stock_data['Close'].astype(float)

    # 计算对数收益率
    stock_data['Log_Return'] = np.log(stock_data['Close_Float'] / stock_data['Close_Float'].shift(1))
    # 计算每日涨跌幅（百分比变化）
    stock_data['Daily_PCT_Change'] = stock_data['Close_Float'].pct_change() * 100
    # 删除多余的列
    stock_data.drop(['Close_Float'], axis=1, inplace=True)


def stock_window_feature(merged_data: pd.DataFrame, window_size: int = 20, long_window: int = 90,
                         stock_code: str = None):
    """
    对股票进行窗口划分同时提取各个窗口下的均值和波动性，同期起到对数据的平滑作用原地修改传入的 DataFrame

    分别计算三种窗口特征：
      1. 滚动窗口计算（重叠）：利用 rolling 方法对 Close 列计算每个滑动窗口内的均值和标准差，
         当窗口内数据不足 window_size 个时返回 NaN。
      2. 非重叠窗口计算：将数据分成不重叠的组，每组包含 window_size 个数据点，
         利用 group by 计算各组的均值和标准差，并将结果映射到 DataFrame 对应行上。
      3. 长期滚动窗口计算：使用较长的 滚动窗口（如 long_window,100）计算长期均值和标准差，
         用以捕捉数据的长期趋势。
    新增列：
      - 'Mean'：滚动窗口内 Close 列的均值
      - 'Volatility'：滚动窗口内 Close 列的标准差（波动性）
      - 'Mean_non_overlap'：非重叠窗口内 Close 列的均值
      - 'EMA':短期滑动窗口内的指数平均，对噪声敏感
      - 'Volatility_non_overlap'：非重叠窗口内 Close 列的标准差
      - 'Long_Mean'：长期滚动窗口内 Close 列的均值
      - 'Long_Volatility'：长期滚动窗口内 Close 列的标准差
    """
    # 1. 滚动窗口计算（短期）：计算窗口内的均值、指数加权平均和标准差（波动性）
    merged_data['Mean'] = merged_data['Close'].rolling(window=window_size, min_periods=window_size).mean()
    merged_data['EMA'] = merged_data['Close'].ewm(span=window_size, adjust=False).mean()
    merged_data['Volatility'] = merged_data['Close'].rolling(window=window_size, min_periods=window_size).std()

    # 2. 非重叠窗口计算：将数据分成每 window_size 个一组，计算各组均值和标准差
    merged_data['group'] = np.arange(len(merged_data)) // window_size
    merged_data['Mean_non_overlap'] = merged_data.groupby('group')['Close'].transform('mean')
    merged_data['Volatility_non_overlap'] = merged_data.groupby('group')['Close'].transform('std')
    merged_data.drop(columns='group', inplace=True)

    # 3. 长期滚动窗口计算：利用较长的窗口（long_window default 90）计算
    merged_data['Long_Mean'] = merged_data['Close'].rolling(window=long_window, min_periods=long_window).mean()
    merged_data['Long_Volatility'] = merged_data['Close'].rolling(window=long_window, min_periods=long_window).std()

    """
    绘制股票数据的各项特征指标：
      - 第一子图：价格及移动均值（短期 rolling 均值、非重叠均值、长期 rolling 均值）
      - 第二子图：波动性，各窗口的标准差特征（短期 rolling 标准差、非重叠标准差、长期 rolling 标准差）
    """
    plt.figure(figsize=(14, 10))

    # 子图1: 绘制价格和均值特征
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(merged_data.index, merged_data['Close'], label='Close Price', color='black')
    ax1.plot(merged_data.index, merged_data['Mean'], label='Short-term Mean (20)', linestyle='--', color='blue')
    ax1.plot(merged_data.index, merged_data['EMA'], label='Short-term EMA (20)', linestyle='-.', color='orange')
    ax1.plot(merged_data.index, merged_data['Mean_non_overlap'], label='Non-overlap Mean (20)', linestyle='-.',
             color='green')
    ax1.plot(merged_data.index, merged_data['Long_Mean'], label='Long-term Mean (90)', linestyle=':', color='red')

    ax1.set_title('Price and Moving Averages')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()

    # 子图2: 绘制波动性特征
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(merged_data.index, merged_data['Volatility'], label='Short-term Volatility (20)', linestyle='--',
             color='blue')
    ax2.plot(merged_data.index, merged_data['Volatility_non_overlap'], label='Non-overlap Volatility (20)',
             linestyle='-.', color='green')
    ax2.plot(merged_data.index, merged_data['Long_Volatility'], label='Long-term Volatility (90)', linestyle=':',
             color='red')

    ax2.set_title('Volatility Features')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'../picture/{stock_code}/stock_window_features.svg', bbox_inches='tight')
    plt.show()
    plt.close()


def macro_window_feature(merged_data: pd.DataFrame, stock_code: str = None)->pd.DataFrame:
    """
    处理宏观数据特征（同比、环比、趋势分解等）
    参数：
        merged_data: 包含股票和宏观数据的DataFrame（索引为日期，已对齐到交易日）
        macro_cols: 需要处理的宏观指标列名（如['CPI', 'PPI', 'PMI']）
        lag_days: 宏观数据发布延迟天数（默认22天，约1个月）

    返回：
        处理后的DataFrame，新增特征列
    """
    macro_cols = ['CPI', 'PPI', 'PMI']

    # 1. 处理数据延迟（避免未来信息）

    # 2. 添加自然年月列
    merged_data['Year'] = merged_data.index.year
    merged_data['Month'] = merged_data.index.month

    # 3. 对每个宏观指标计算特征
    for col in macro_cols:
        # 3.1 同比（YoY）：当前月 vs 去年同月
        merged_data[f'{col}_LastYear'] = merged_data.groupby('Month')[col].shift(12)  # 12个月前的同月数据
        merged_data[f'{col}_YoY'] = (merged_data[col] / merged_data[f'{col}_LastYear'] - 1) * 100

        # 3.2 环比（MoM）：当前月 vs 上月
        # 先按年月排序确保正确shift
        merged_data_sorted = merged_data.sort_values(['Year', 'Month'])
        merged_data_sorted[f'{col}_LastMonth'] = merged_data_sorted.groupby('Year')[col].shift(1)
        merged_data = merged_data_sorted.sort_index()  # 恢复原始索引顺序
        merged_data[f'{col}_MoM'] = (merged_data[col] / merged_data[f'{col}_LastMonth'] - 1) * 100

        # 3.3 趋势分解（Hodrick-Prescott Filter）
        cycle, trend = hpfilter(merged_data[col].dropna(), lamb=14400)
        merged_data[f'{col}_Trend'] = trend
        merged_data[f'{col}_Cycle'] = cycle

        # 滚动窗口 = 12个月（自然月跨度），min_periods可设为1（因每月至少有一个数据点）
        merged_data[f'{col}_12M_Mean'] = merged_data[col].rolling(window='252D', min_periods=1).mean()
        merged_data[f'{col}_12M_Vol'] = merged_data[col].rolling(window='252D', min_periods=1).std()

    # 5. 清理中间列
    merged_data.drop(columns=['Year', 'Month'] +
                             [f'{col}_LastYear' for col in macro_cols] +
                             [f'{col}_LastMonth' for col in macro_cols],
                     inplace=True)

    print("Existing columns:", merged_data.columns.tolist())

    # 6. 绘制特征图像（CPI）
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 图1：CPI 原始数据与 12 个月均值
    axes[0].plot(merged_data.index, merged_data['CPI'], label='CPI', color='blue', alpha=0.6)
    axes[0].plot(merged_data.index, merged_data['CPI_12M_Mean'], label='12M Mean', color='orange', linewidth=2)
    axes[0].set_title('CPI and 12-Month Rolling Mean')
    axes[0].legend()

    # 图2：CPI YoY 变化率
    axes[1].plot(merged_data.index, merged_data['CPI_YoY'], label='CPI YoY (%)', color='green')
    axes[1].set_title('CPI Year-over-Year Change (%)')
    axes[1].legend()

    # 图3：CPI MoM 变化率
    axes[2].plot(merged_data.index, merged_data['CPI_MoM'], label='CPI MoM (%)', color='red')
    axes[2].set_title('CPI Month-over-Month Change (%)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'../picture/{stock_code}/macro_window_features.svg', bbox_inches='tight')
    plt.show()
    plt.close()
    return merged_data

def lag_cross_feature(merged_data: pd.DataFrame):
    """
    为股票数据和宏观数据构造滞后特征
    此函数直接原地修改传入的 stock_data，不返回新对象。
    """
    # 将关键列转换为 float（如果它们是 Decimal 类型）
    cols_to_convert = ['Close', 'Volatility', 'CPI_12M_Vol', 'Long_Mean', 'CPI_Trend',
                       'CPI_YoY', 'PMI', 'PMI_Trend', 'Volume']
    for col in cols_to_convert:
        if col in merged_data.columns:
            try:
                merged_data[col] = merged_data[col].astype(float)
            except Exception as e:
                print(f"转换列 {col} 出错：{e}")

    # 宏观数据缺失值处理策略（这里不再处理缺失，只构造滞后特征）
    macro_cols = ['CPI', 'PPI', 'PMI']

    # 处理数据延迟：对宏观数据采用 22 天滞后
    for col in macro_cols:
        merged_data[f"lag_{col}"] = merged_data[col].shift(22)

    # 计算股票价格滞后特征：滞后 5,10,22 天的收盘价
    merged_data['Lag_5'] = merged_data['Close'].shift(5)
    merged_data['Lag_10'] = merged_data['Close'].shift(10)
    merged_data['Lag_22'] = merged_data['Close'].shift(22)

    # 收益率计算（百分比）：(当日收盘价/前日收盘价 - 1)*100
    merged_data['Return'] = merged_data['Close'].pct_change().mul(100).replace([float('inf'), -float('inf')], None)

    # 波动性关联
    merged_data['Volatility_Spread'] = merged_data['Volatility'] - merged_data['CPI_12M_Vol']
    merged_data['Vol_PMI_Corr'] = merged_data['Volatility'].rolling(252, closed='left').corr(merged_data['PMI'])

    # 趋势背离
    merged_data['Trend_Inflation_Spread'] = merged_data['Long_Mean'] - merged_data['CPI_Trend']

    # 宏观经济状态
    merged_data['High_Inflation'] = (merged_data['CPI_YoY'] > 3).astype(int)
    merged_data['Economic_Expansion'] = ((merged_data['PMI'] > 50) &
                                         (merged_data['PMI_Trend'].diff() > 0)).astype(int)

    # 实际价格
    merged_data['Real_Price'] = merged_data['Close'] / (1 + merged_data['CPI_YoY'] / 100)

    # 价格-成交量背离：5日收盘涨幅与成交量5日变化率的差值
    merged_data['Price_Volume_Divergence'] = (
            merged_data['Close'].pct_change(5) -
            merged_data['Volume'].pct_change(5).abs()
    )

    # PMI-价格相关性（滚动1年，252个交易日）
    merged_data['PMI_Price_Corr'] = merged_data['Close'].rolling(252).corr(merged_data['PMI'])

    # 滞胀风险标识：当 CPI_YoY 大于该列75分位数且 PMI 小于 50 时标识为1
    merged_data['Stagflation_Risk'] = ((merged_data['CPI_YoY'] > merged_data['CPI_YoY'].quantile(0.75))
                                       & (merged_data['PMI'] < 50)).astype(int)


# 归一化处理
def normalize_data(merged_data: pd.DataFrame, stock_code: str):
    """
    对输入的 DataFrame 中的所有数值型列进行归一化处理，使得每一列的均值为 0，方差为 1。

    参数:
      data: 待归一化的 DataFrame
    """
    # 选定要归一化的特征

    features_to_normalize = ['Close', 'Volume', 'Volatility', 'CPI', 'PPI', 'PMI']

    # 创建 Min-Max Scaler
    scaler = MinMaxScaler()

    # 对选定的特征进行归一化
    df_normalized = merged_data.copy()  # 保留原数据集
    df_normalized[features_to_normalize] = scaler.fit_transform(merged_data[features_to_normalize])

    # 可视化归一化后的收盘价与交易量
    plt.figure(figsize=(8, 6))

    # 归一化后的收盘价曲线
    plt.subplot(2, 1, 1)
    plt.plot(df_normalized.index, df_normalized['Close'], label='Normalized Close Price', color='blue')
    plt.title('Normalized Close Price')
    plt.xlabel('Date')
    plt.ylabel('Normalized Close Price')
    plt.legend()

    # 归一化后的交易量曲线
    plt.subplot(2, 1, 2)
    plt.plot(df_normalized.index, df_normalized['Volume'], label='Normalized Volume', color='green')
    plt.title('Normalized  Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Normalized Volume')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'../picture/{stock_code}/normalized_features_plot.svg')
    plt.show()
    plt.close()


async def main():
    await database.connect()
    await feature_engineering(stock_code=ExponentEnum.SZCZ.get_code(), start_date=None, end_date=None)
    await database.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
