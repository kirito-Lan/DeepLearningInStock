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
from manager.decoration.dbconnect import db_connection
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
    # 2.宏观数据特征
    macro_feature(merged_data=merged_data, stock_code=stock_code)
    # 股票价格特征
    #stock_feature(merged_data=merged_data, stock_code=stock_code)
    # 季节特征
    seasonal_feature(merged_data=merged_data)
    # 交易量异常值修复
    volume_abnormal_feature(stock_data=merged_data, stock_code=stock_code)
    # 股票和宏观数据交叉特征
    cross_feature(merged_data=merged_data)
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
    # 将cloums=['Open','Close', 'High', 'Low', 'Volume','CPI', 'PPI', 'PMI']全部转换成float
    for col in ['Open', 'Close', 'High', 'Low', 'Volume', 'CPI', 'PPI', 'PMI']:
        merged_data[col] = merged_data[col].astype(float)
    # 返回合并后的数据
    return merged_data


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
    #plt.show()  # 显示图像后再关闭图表
    plt.close()

# 股票季节分解
def seasonal_feature(merged_data: pd.DataFrame):
    # 对股票收盘价数据进行季节性分解
    decomposition = seasonal_decompose(merged_data['Close'], model='multiplicative', period=252)
    trend = decomposition.trend  # 趋势
    seasonality = decomposition.seasonal  # 季节性
    residual = decomposition.resid  # 残差
    merged_data['Close_Trend'] = trend
    merged_data['Close_Seasonal'] = seasonality
    merged_data['Close_Residual'] = residual

# 全部关于股票的特征
def stock_feature(merged_data: pd.DataFrame, short_windows: int = 20, long_window: int = 60,
                         stock_code: str = None):


    # 收益率计算（百分比）：(当日收盘价/前日收盘价 - 1)*100
    #merged_data['Return'] = merged_data['Close'].pct_change().mul(100).replace([float('inf'), -float('inf')], None)
    # 计算对数收益率
    #merged_data['Log_Return'] = np.log(merged_data['Close'] / merged_data['Close'].shift(1))
    # 计算每日涨跌幅（百分比变化）
    merged_data['Daily_PCT_Change'] = merged_data['Close'].pct_change() * 100

    # 1. 滚动窗口计算（短期）：计算窗口内的均值、指数加权平均和标准差（波动性）
    #merged_data['Mean'] = merged_data['Close'].rolling(window=short_windows, min_periods=short_windows).mean()
    merged_data['EMA'] = merged_data['Close'].ewm(span=short_windows, adjust=False).mean()
    merged_data['Volatility'] = merged_data['Close'].rolling(window=short_windows, min_periods=short_windows).std()

    # 3. 长期滚动窗口计算：利用较长的窗口（long_window default 60）计算
    merged_data['Long_Mean'] = merged_data['Close'].rolling(window=long_window, min_periods=long_window).mean()
    merged_data['Long_Volatility'] = merged_data['Close'].rolling(window=long_window, min_periods=long_window).std()

    # 计算股票价格滞后特征：滞后 5,10,22 天的收盘价
    # merged_data['Lag_10'] = merged_data['Close'].shift(10)
    merged_data['Lag_22'] = merged_data['Close'].shift(22)
    merged_data['Lag_60'] = merged_data['Close'].shift(60)

    """
    绘制股票数据的各项特征指标：
      - 第一子图：价格及移动均值（短期 rolling 均值、长期 rolling 均值）
      - 第二子图：波动性，各窗口的标准差特征（短期 rolling 标准差、长期 rolling 标准差）
    """
    # # region
    # plt.figure(figsize=(14, 10))
    # # 子图1: 绘制价格和均值特征
    # ax1 = plt.subplot(2, 1, 1)
    # ax1.plot(merged_data.index, merged_data['Close'], label='Close Price', color='black')
    # ax1.plot(merged_data.index, merged_data['Mean'], label=f'Short-term Mean ({short_windows})', linestyle='--', color='blue')
    # ax1.plot(merged_data.index, merged_data['EMA'], label=f'Short-term EMA ({short_windows})', linestyle='-.', color='orange')
    # ax1.plot(merged_data.index, merged_data['Long_Mean'], label=f'Long-term Mean ({long_window})', linestyle=':', color='red')
    #
    # ax1.set_title('Price and Moving Averages')
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Price')
    # ax1.legend()
    #
    # # 子图2: 绘制波动性特征
    # ax2 = plt.subplot(2, 1, 2)
    # ax2.plot(merged_data.index, merged_data['Volatility'], label='Short-term Volatility (20)', linestyle='--',
    #          color='blue')
    # ax2.plot(merged_data.index, merged_data['Long_Volatility'], label='Long-term Volatility (90)', linestyle=':',
    #          color='red')
    #
    # ax2.set_title('Volatility Features')
    # ax2.set_xlabel('Time')
    # ax2.set_ylabel('Volatility')
    # ax2.legend()
    #
    # plt.tight_layout()
    # plt.savefig(f'../picture/{stock_code}/stock_window_features.svg', bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # #endregion

# 全部关于宏观数据的特征
def macro_feature(merged_data: pd.DataFrame, stock_code: str = None):

    macro_cols = ['CPI', 'PPI', 'PMI']

    # 计算波动性（30天滚动标准差）
    for column in macro_cols:
        merged_data[f'{column}_Volatility'] = merged_data[column].rolling(window=30).std()

    # 3. 对每个宏观指标计算特征
    for col in macro_cols:
        if col == 'PMI':
            merged_data[f'{col}_MOM'] = merged_data[col].pct_change() * 100
        else:
            merged_data[f'{col}_MOM'] = merged_data[col]-100

    # 趋势分解（Hodrick-Prescott Filter）
    for col in macro_cols:
        cycle, trend = hpfilter(merged_data[col].dropna(), lamb=14400)
        merged_data[f'{col}_Trend'] = trend
        merged_data[f'{col}_Cycle'] = cycle


# 交叉特征
def cross_feature(merged_data: pd.DataFrame):
    """
    为股票数据和宏观数据构造滞后特征
    此函数直接原地修改传入的 stock_data，不返回新对象。
    """
    # 宏观经济状态
    merged_data['High_Inflation'] = (merged_data['CPI_MOM'] > 2).astype(int)
    merged_data['Economic_Expansion'] = ((merged_data['PMI'] > 50) &
                                         (merged_data['PMI_Trend'].diff() > 0)).astype(int)

    # PMI-价格相关性（滚动1年，252个交易日）
    merged_data['PMI_Price_Corr'] = merged_data['Close'].rolling(252).corr(merged_data['PMI'])

    # 滞胀风险标识：当 CPI_MoM 大于该列75分位数且 PMI 小于 50 时标识为1
    merged_data['Stagflation_Risk'] = ((merged_data['CPI_MOM'] > merged_data['CPI_MOM'].quantile(0.75))
                                       & (merged_data['PMI'] < 50)).astype(int)

    # 1. 整合宏观经济趋势：计算 CPI, PPI, PMI 趋势的均值
    merged_data['Macro_Trend'] = merged_data[['CPI_Trend', 'PPI_Trend', 'PMI_Trend']].mean(axis=1)
    # 2. 构造背离趋势特征：计算股价趋势与综合宏观趋势之间的差
    merged_data['Divergence_Trend'] = merged_data['Close_Trend'] - merged_data['Macro_Trend']



@db_connection
async def main():
    start,end = format_date(None, None)
    await feature_engineering(stock_code=ExponentEnum.HS300.get_code(), start_date=start,end_date=end)


if __name__ == '__main__':
    asyncio.run(main())
