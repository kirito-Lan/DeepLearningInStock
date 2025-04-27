# 股票特征工程
import asyncio
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
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
from model.entity.MacroData import MacroData
from utils.ReFormatDate import format_date


# 股票数据的特征工程
async def feature_engineering(stock_code: str, start_date: str, end_date: str):

    # 获取合并的数据
    merged_data= await get_merged_data(end_date, start_date, stock_code)

    # #时间序列分解
    # decompose_time_series(stock_data=stock_data)
    # # 交易量的特征
    # volume_feature(stock_data=stock_data)
    # # 窗口函数计算长期和短期的均值和波动性
    # window_feature(stock_data=stock_data)



    pass


async def get_merged_data(end_date, start_date, stock_code):
    """
    获取合并并且处理后股票和宏观数据的数据
    """
    # 获取股票数据
    start_date, end_date = format_date(start_date=start_date, end_date=end_date)
    stock_data = await StockDataManage.get_stock_data(stock_code=stock_code, start_date=start_date, end_date=end_date)
    stock_data["trade_date"] = pd.to_datetime(stock_data['trade_date'], format='%Y-%m-%d', errors='coerce')
    stock_data = stock_data.set_index('trade_date')
    # 数据库获取的CPI数据是月率MOM  PPI是年率 YOY PMI是原始数据需要计算的话需要进行转换恢复原始值
    start = stock_data.index[0]
    end = stock_data.index[-1]
    log.info(f"获取到数据库最新的股票起止时间：Start: {start}, End: {end}")
    cpi_data = await MacroDataManage.get_macro_data(types=MacroDataEnum.CPI, start_date=start, end_date=end)
    cpi_data = cpi_data.set_index('report_date')
    ppi_data = await MacroDataManage.get_macro_data(types=MacroDataEnum.PPI, start_date=start, end_date=end)
    ppi_data = ppi_data.set_index('report_date')
    pmi_data = await MacroDataManage.get_macro_data(types=MacroDataEnum.PMI, start_date=start, end_date=end)
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
        merged.drop(columns=['CPI_MOM', 'PPI_YOY'], axis=1,inplace=True)
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
    decomposition = seasonal_decompose(stock_data['Close'], model='multiplicative', period=252)
    trend = decomposition.trend  # 趋势
    seasonality = decomposition.seasonal  # 季节性
    residual = decomposition.resid  # 残差
    stock_data['Trend'] = trend
    stock_data['Seasonal'] = seasonality
    stock_data['Residual'] = residual
    # 计算波动性（20天滚动标准差）
    stock_data['Volatility'] = stock_data['Close'].rolling(window=20).std()


# 股票的交易量特征和异常值检测修复
def volume_feature(stock_data: pd.DataFrame):
    """
    计算股票的交易量特征,交易量的变化率
    处理异常的交易数据 Z-Score方法标记
    同时使用四分位法寻找异常值，然后用线性插值修复
    Args:
        stock_data: 包含股票交易量数据的DataFrame
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
    plt.savefig('./picture/volume_correction.svg')
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
    # 计算对数收益率
    stock_data['Log_Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    # 计算每日涨跌幅
    stock_data['Daily_PCT_Change'] = stock_data['Close'].pct_change()


def window_feature(stock_data: pd.DataFrame, window_size: int = 20, long_window: int = 100):
    """
    对股票进行窗口划分同时提取各个窗口下的均值和波动性，同期起到对数据的平滑作用原地修改传入的 DataFrame

    分别计算三种窗口特征：
      1. 滚动窗口计算（重叠）：利用 rolling 方法对 Close 列计算每个滑动窗口内的均值和标准差，
         当窗口内数据不足 window_size 个时返回 NaN。
      2. 非重叠窗口计算：将数据分成不重叠的组，每组包含 window_size 个数据点，
         利用 groupby 计算各组的均值和标准差，并将结果映射到 DataFrame 对应行上。
      3. 长期滚动窗口计算：使用较长的滚动窗口（如 long_window，默认200）计算长期均值和标准差，
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
    stock_data['Short_Mean'] = stock_data['Close'].rolling(window=window_size, min_periods=window_size).mean()
    stock_data['EMA'] = stock_data['Close'].ewm(span=window_size, adjust=False).mean()
    stock_data['Short_Volatility'] = stock_data['Close'].rolling(window=window_size, min_periods=window_size).std()

    # 2. 非重叠窗口计算：将数据分成每 window_size 个一组，计算各组均值和标准差
    stock_data['group'] = np.arange(len(stock_data)) // window_size
    stock_data['Mean_non_overlap'] = stock_data.groupby('group')['Close'].transform('mean')
    stock_data['Volatility_non_overlap'] = stock_data.groupby('group')['Close'].transform('std')
    stock_data.drop(columns='group', inplace=True)

    # 3. 长期滚动窗口计算：利用较长的窗口（long_window，默认200）计算
    stock_data['Long_Mean'] = stock_data['Close'].rolling(window=long_window, min_periods=long_window).mean()
    stock_data['Long_Volatility'] = stock_data['Close'].rolling(window=long_window, min_periods=long_window).std()

    """
    绘制股票数据的各项特征指标：
      - 第一子图：价格及移动均值（短期 rolling 均值、非重叠均值、长期 rolling 均值）
      - 第二子图：波动性，各窗口的标准差特征（短期 rolling 标准差、非重叠标准差、长期 rolling 标准差）
    """
    plt.figure(figsize=(14, 10))

    # 子图1: 绘制价格和均值特征
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(stock_data.index, stock_data['Close'], label='Close Price', color='black')
    ax1.plot(stock_data.index, stock_data['Mean'], label='Short-term Mean (20)', linestyle='--', color='blue')
    ax1.plot(stock_data.index, stock_data['EMA'], label='Short-term EMA (20)', linestyle='-.', color='orange')
    ax1.plot(stock_data.index, stock_data['Mean_non_overlap'], label='Non-overlap Mean (20)', linestyle='-.',
             color='green')
    ax1.plot(stock_data.index, stock_data['Long_Mean'], label='Long-term Mean (200)', linestyle=':', color='red')

    ax1.set_title('Price and Moving Averages')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()

    # 子图2: 绘制波动性特征
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(stock_data.index, stock_data['Volatility'], label='Short-term Volatility (20)', linestyle='--',
             color='blue')
    ax2.plot(stock_data.index, stock_data['Volatility_non_overlap'], label='Non-overlap Volatility (20)',
             linestyle='-.', color='green')
    ax2.plot(stock_data.index, stock_data['Long_Volatility'], label='Long-term Volatility (200)', linestyle=':',
             color='red')

    ax2.set_title('Volatility Features')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('./picture/window_features.svg', bbox_inches='tight')
    plt.show()
    plt.close()


def lag_cross_feature(merged: pd.DataFrame):
    """
    为股票数据和宏观数据构造滞后特征
    此函数直接原地修改传入的 stock_data，不返回新对象。
    """


    #  宏观数据缺失值处理策略
    # 1. 对宏观数据使用前向填充（假设数据发布后持续有效）
    macro_cols = ['CPI', 'PPI', 'PMI', 'CPI_MOM', 'PPI_YOY']
    merged[macro_cols] = merged[macro_cols].fillna(method='ffill')

    # 计算滞后特征，分别为滞后1天、5天、10天的收盘价
    merged['Lag_1'] = merged['Close'].shift(1)
    merged['Lag_5'] = merged['Close'].shift(5)
    merged['Lag_10'] = merged['Close'].shift(10)

    # 线性插值填充 NaN
    merged['Lag_1'].interpolate(method='linear', inplace=True)
    merged['Lag_5'].interpolate(method='linear', inplace=True)
    merged['Lag_10'].interpolate(method='linear', inplace=True)

    # 收益率计算（百分比）：(当日收盘价/前日收盘价 - 1)*100
    merged['Return'] = merged['Close'].pct_change().mul(100).replace([float('inf'), -float('inf')], None)

    # 宏观交叉特征（增加有效性检查）
    required_macro = {
        'Return_CPI_gap': ['Return', 'CPI_MOM'],
        'Return_PPI_gap': ['Return', 'PPI_YOY'],
        'PMI_change': ['PMI']
    }
    for feat, deps in required_macro.items():
        if all(col in merged for col in deps):
            if feat == 'PMI_change':
                merged[feat] = merged['PMI'].pct_change()
            else:
                merged[feat] = merged[deps[0]] - merged[deps[1]]
        else:
            print(f"警告：无法创建特征 {feat}，缺少依赖列 {deps}")







def macroeconomic_feature(merged_data: pd.DataFrame):
    # FIXME 这里修改为传入已经和股票数据合并的数据 时间频率已经是日了 修正代码逻辑
    """
    改进后的宏观特征工程，主要优化：
    1. 基于业务逻辑的滚动窗口
    2. 正确的季节性处理
    3. 动态特征有效性检测
    """
    # 滞后特征（动态检测可用列）
    for col in merged_data:
        for lag in [1, 3, 12]:  # 月、季、年滞后
            if f"{col}_lag{lag}" not in merged_data:
                merged_data[f"{col}_lag{lag}"] = merged_data[col].shift(lag)

    # 滚动特征（适应宏观数据频率）
    windows = {
        'monthly': 21,  # 约1个月交易日
        'quarterly': 63,  # 约3个月
        'annual': 252  # 约1年
    }
    for col in merged_data:
        for win_name, win_size in windows.items():
            sma_col = f"{col}_SMA_{win_name}"
            std_col = f"{col}_STD_{win_name}"

            if sma_col not in merged_data:
                merged_data[sma_col] = merged_data[col].rolling(win_size, min_periods=1).mean()
            if std_col not in merged_data:
                merged_data[std_col] = merged_data[col].rolling(win_size, min_periods=1).std()

    # 季节性分解（仅适用于足够长的序列）
    if len(merged_data) > 3 * 365:  # 至少3年数据
        try:
            from statsmodels.tsa.seasonal import STL  # 更鲁棒的分解方法

            for col in ['CPI', 'PPI']:
                # 使用STL分解处理复杂季节模式
                stl = STL(
                    merged_data[col].interpolate(),
                    period=365,  # 年周期
                    robust=True
                )
                res = stl.fit()

                merged_data[f"{col}_trend"] = res.trend
                merged_data[f"{col}_seasonal"] = res.seasonal
                merged_data[f"{col}_resid"] = res.resid
        except ImportError:
            print("建议安装statsmodels以获得完整季节分解功能")

    # 日期特征（考虑节假日影响）
    merged_data['day_of_month'] = merged_data.index.day
    merged_data['is_month_end'] = merged_data.index.is_month_end.astype(int)

    # 季度哑变量
    merged_data['quarter'] = merged_data.index.quarter
    macro_data = pd.get_dummies(merged_data, columns=['quarter'], prefix='qtr')


# 归一化处理
def normalize_data(data: pd.DataFrame):
    """
    对输入的 DataFrame 中的所有数值型列进行归一化处理，使得每一列的均值为 0，方差为 1。

    参数:
      data: 待归一化的 DataFrame
    """


async def main():
    await database.connect()
    await feature_engineering(stock_code=ExponentEnum.SZCZ.get_code(), start_date=None, end_date=None)
    await database.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
