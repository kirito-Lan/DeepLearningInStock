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
async def stock_feature_engineering(stock_code: str, start_date: str, end_date: str):
    # 获取股票数据
    start_date, end_date = format_date(start_date=start_date, end_date=end_date)
    stock_data = await StockDataManage.get_stock_data(stock_code=stock_code, start_date=start_date, end_date=end_date)
    stock_data["trade_date"] = pd.to_datetime(stock_data['trade_date'], format='%Y-%m-%d', errors='coerce')
    stock_data = stock_data.set_index('trade_date')
    # FIXME 数据库获取的CPI数据是月率MOM  PPI是年率 YOY PMI是原始数据需要计算的话需要进行转换恢复原始值
    start = stock_data.index[0]
    end = stock_data.index[-1]
    log.info(f"Start: {start}, End: {end}")
    cpi_data = await MacroDataManage.get_macro_data(types=MacroDataEnum.CPI, start_date=start, end_date=end)
    cpi_data = cpi_data.set_index('report_date')
    ppi_data = await MacroDataManage.get_macro_data(types=MacroDataEnum.PPI, start_date=start, end_date=end)
    ppi_data = ppi_data.set_index('report_date')
    pmi_data = await MacroDataManage.get_macro_data(types=MacroDataEnum.PMI, start_date=start, end_date=end)
    pmi_data = pmi_data.set_index('report_date')
    # 计算原始cpi ppi数据
    cpi_data['CPI'] = cpi_data['current_value'].apply(compute_index)
    ppi_data['PPI'] = ppi_data['current_value'].apply(compute_index)
    # 重置列名
    cpi_data.columns = ['CPI_MOM','CPI']
    ppi_data.columns = ['PPI_YOY','PPI']
    pmi_data.columns = ['PMI']
    stock_data.columns = ['Open','Close','High','Low','Volume']

    #时间序列分解
    decompose_time_series(stock_data=stock_data)
    # 交易量的特征
    volume_feature(stock_data=stock_data)
    # 窗口函数计算长期和短期的均值和波动性
    window_feature(stock_data=stock_data)


    # print("cpi_data:\n{}".format(cpi_data))
    # print("ppi_data:\n{}".format(ppi_data))
    # print("pmi_data:\n{}".format(pmi_data))
    print("stock_data:\n{}".format(stock_data))


    pass


def compute_index(x):
    # 根据公式：指数 = 100 / (1 - current_value / 100)
    result = Decimal('100')+Decimal(x)
    # 舍入到两位小数（half-up）
    return result.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)

# 时间序列分解 seasonal_decompose和 波动计算
def decompose_time_series(stock_data:pd.DataFrame):
    """
    对时间序列进行分解，并绘制分解结果
    以252个股票交易日为基准，使用乘法模型计算
    同时计算20天的滚动标准差作为波动性特征
    Args:
        stock_data: 待分解的时间序列数据
    Returns:
        None
    """
    decomposition = seasonal_decompose(stock_data['Close'], model='multiplicative', period=252)
    trend = decomposition.trend  # 趋势
    seasonality = decomposition.seasonal # 季节性
    residual = decomposition.resid # 残差
    stock_data['Trend'] = trend
    stock_data['Seasonal'] = seasonality
    stock_data['Residual'] = residual
    # 计算波动性（20天滚动标准差）
    stock_data['Volatility'] = stock_data['Close'].rolling(window=20).std()

# 股票的交易量特征和异常值检测修复
def volume_feature(stock_data:pd.DataFrame):
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
def reward_rate_feature(stock_data:pd.DataFrame):
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

# 宏观经济特征
# FIXME 宏观经济和股票数据特征提取
def macroeconomic_feature(macro_data:pd.DataFrame):
    """
    计算宏观经济特征
    Args:
        macro_data: 包含宏观经济数据的DataFrame
    """
    pass



def window_feature(stock_data: pd.DataFrame, window_size: int = 20, long_window: int = 200):
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
    stock_data['Mean'] = stock_data['Close'].rolling(window=window_size, min_periods=window_size).mean()
    stock_data['EMA'] = stock_data['Close'].ewm(span=window_size, adjust=False).mean()
    stock_data['Volatility'] = stock_data['Close'].rolling(window=window_size, min_periods=window_size).std()

    # 2. 非重叠窗口计算：将数据分成每 window_size 个一组，计算各组均值和标准差
    stock_data['group'] = np.arange(len(stock_data)) // window_size
    stock_data['Mean_non_overlap'] = stock_data.groupby('group')['Close'].transform('mean')
    stock_data['Volatility_non_overlap'] = stock_data.groupby('group')['Close'].transform('std')
    stock_data.drop(columns='group', inplace=True)

    # 3. 长期滚动窗口计算：利用较长的窗口（例如 long_window，默认200）计算
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
    plt.savefig('./picture/window_features.svg',bbox_inches='tight')
    plt.show()
    plt.close()

# 数据平滑处理
def smooth_data(data: pd.Series, window_size: int = 20):
    pass


async def main():
    await database.connect()
    await stock_feature_engineering(stock_code=ExponentEnum.SZCZ.get_code(), start_date=None, end_date=None)
    await database.disconnect()


if __name__ == '__main__':
    asyncio.run(main())