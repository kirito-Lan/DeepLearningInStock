import asyncio

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

from config.LoguruConfig import log
from constant.ExponentEnum import ExponentEnum
from constant.MaroDataEnum import MacroDataEnum
from manager import StockDataManage, MacroDataManage
from model.entity.BaseMeta.BaseMeta import database
from utils.ReFormatDate import format_date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import seaborn as sns


async def analyse_stock_data(stock_code, start_date, end_date):
    # 获取到股票的数据
    start_date, end_date = format_date(start_date=start_date, end_date=end_date)
    stock_data = await StockDataManage.get_stock_data(stock_code=stock_code, start_date=start_date, end_date=end_date)
    # 转换时间格式
    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'], errors='coerce', format="%Y-%m-%d")
    # 将date列设置为索引
    stock_data.set_index('trade_date', inplace=True)
    # 基本信息图
    await stock_basic_info(stock_code, stock_data)
    # 股票的趋势与季节性分解
    await stock_trend_seasonal(stock_code,stock_data)
    # 股票波动分析
    await stock_volatility(stock_code,stock_data)
    # 异常值检测
    await abnormal_value_detection(stock_code,stock_data)
    # 股票与宏观数据的相关性
    await stock_macro_correlation(stock_code=stock_code, stock_data=stock_data)
    #研究交易量和收盘价的相关性
    await stock_volume_price_correlation(stock_code=stock_code, stock_data=stock_data)


async def stock_basic_info(stock_code, stock_data):
    """
    分析股票的基本信息
    1.股票的时间和交易量图
    2.股票的时间和收盘价
    3.股票的收盘价的概率密度分布图
    """

    # 股票的时间和交易量图
    plt.figure()
    plt.title(f'{stock_code}-Volume&Date')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.plot(stock_data.index, stock_data['volume'], '-', label='close_price Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()
    # 股票的时间和收盘价
    plt.figure()
    plt.title(f'{stock_code}- Date&close_price Price')
    plt.xlabel('Date')
    plt.ylabel('close_price Price')
    plt.plot(stock_data.index, stock_data['close_price'], '-', label='close_price Price')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()  # 清除当前图像
    # 股票的收盘价的概率密度分布图
    plt.figure(figsize=(8, 5))
    sns.kdeplot(stock_data['close_price'], fill=True, color='b')
    plt.title(f'{stock_code} - close_price Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Probability Density')
    plt.tight_layout()
    plt.show()
    plt.close()


async def stock_trend_seasonal(stock_code, stock_data):
    """
    股票的趋势与季节性分解
    """
    # 提取收盘价数据
    close_series = stock_data['close_price']

    # 进行时间序列分解，使用multiplicative（乘法）模型，周期为365（假设年周期性）
    decomposition = seasonal_decompose(close_series, model='multiplicative', period=365)

    # 绘制趋势、季节性和残差
    plt.figure(figsize=(10, 6))

    # 创建4行1列的子图，并分别绘制趋势、季节性、残差和观察到的收盘价
    plt.subplot(411)
    plt.plot(close_series, label='Observed')
    plt.title(f'Observed {stock_code} close_price Prices')

    # 绘制趋势
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.title('Trend')

    # 绘制季节性
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.title('Seasonal')

    # 绘制残差
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals')
    plt.title('Residuals')

    plt.tight_layout()
    plt.show()
    plt.savefig('./picture/trend_seasonality_decomposition.svg')
    plt.close()


async def stock_volatility(stock_code, stock_data):
    """
    股票的波动性分析
    """
    # pands.pct_change() 函数计算收益率  当前值/前值-1
    # 1. 计算日收益率
    stock_data['Day_Returns'] = stock_data['close_price'].pct_change()

    # 2. 计算20天移动标准差作为短期波动性指标
    window_size = 20  # 窗口大小为20天
    stock_data['Rolling_Std'] = stock_data['close_price'].rolling(window=window_size).std()

    # 计算年化收益率 用每日收益的滑动标准差 *np.sqrt(252)
    # 年化波动率 = 每日收益率的标准差 × √252
    stock_data['Volatility'] = stock_data['Day_Returns'].rolling(window=window_size).std() * np.sqrt(252)

    # 三个图合一
    plt.figure(figsize=(10, 6))

    # 子图1: 收盘价
    plt.subplot(3, 1, 1)
    plt.plot(stock_data.index, stock_data['close_price'], label='close_price Price', color='blue')
    plt.title(f'{stock_code} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('close_price Price')
    plt.grid(True)
    plt.legend()

    # 子图2: 短期波动性（移动标准差）
    plt.subplot(3, 1, 2)
    plt.plot(stock_data.index, stock_data['Rolling_Std'],
             label='20-Day Rolling Std (Volatility)',
             color='orange')
    plt.title(f'{stock_code}  20-Day Rolling Standard Deviation (Volatility)')
    plt.xlabel('Date')
    plt.ylabel('Rolling Std (Volatility)')
    plt.grid(True)
    plt.legend()

    # 子图3: 年化波动率
    plt.subplot(3, 1, 3)
    plt.plot(stock_data.index, stock_data['Volatility'], label='Annualized Volatility', color='red')
    plt.title(f'{stock_code}  Annualized Volatility')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # 保存图像
    plt.savefig('./picture/volatility_analysis.svg')
    plt.close()


async def abnormal_value_detection(stock_code, stock_data):
    """
    股票异常值分析
    """
    # 1. 使用箱线图检测交易量和收盘价中的异常值
    plt.figure(figsize=(8, 4))
    # 绘制箱线图检测收盘价中的异常值
    plt.subplot(1, 2, 1)
    sns.boxplot(y=stock_data['close_price'], color='lightblue')
    plt.title(f'Box Plot of {stock_code} Closing Price')

    # 绘制箱线图检测交易量中的异常值
    plt.subplot(1, 2, 2)
    sns.boxplot(y=stock_data['volume'], color='lightgreen')
    plt.title(f'Box Plot of {stock_code} Trading Volume')

    # 保存箱线图
    plt.tight_layout()
    plt.savefig('./picture/boxplot_outliers.svg')
    plt.show()
    plt.close()
    # 2. 使用 Z 分数法检测异常值
    # 定义 Z 分数函数，z = (x - mean) / std
    def z_score(df, column_name):
        # 将列转换为浮点类型
        column_data = df[column_name].astype(float)
        # 计算均值和标准差
        mean_value = column_data.mean()
        std_value = column_data.std()
        # 计算Z-score
        return (column_data - mean_value) / std_value

    # 计算收盘价和交易量的 Z 分数
    stock_data['Close_zscore'] = z_score(stock_data, 'close_price')
    stock_data['Volume_zscore'] = z_score(stock_data, 'volume')

    # 设置 Z 分数的阈值，通常超过3或小于-3的值被视为异常值
    z_threshold = 3

    # 识别出异常的收盘价和交易量
    price_outliers = stock_data[abs(stock_data['Close_zscore']) > z_threshold]
    volume_outliers = stock_data[abs(stock_data['Volume_zscore']) > z_threshold]

    # 输出检测到的异常值
    log.info(f"Detected {len(price_outliers)} price outliers and {len(volume_outliers)} volume outliers.")

    # 3. 可视化检测到的异常点
    plt.figure(figsize=(10, 6))

    # 绘制收盘价和异常点
    plt.subplot(2, 1, 1)
    plt.plot(stock_data.index, stock_data['close_price'], label='Closing Price', color='blue')
    plt.scatter(price_outliers.index, price_outliers['close_price'], color='red', label='Price Outliers', zorder=5)
    plt.title(f'{stock_code} Closing Price with Outliers')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()

    # 绘制交易量和异常点
    plt.subplot(2, 1, 2)
    plt.plot(stock_data.index, stock_data['volume'], label='Trading Volume', color='green')
    plt.scatter(volume_outliers.index, volume_outliers['volume'], color='red', label='Volume Outliers', zorder=5)
    plt.title(f'{stock_code} Trading Volume with Outliers')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    # 保存检测到的异常值图
    plt.tight_layout()
    plt.show()
    plt.savefig('./picture/abnormal_detection.svg')
    plt.close()


async def stock_macro_correlation(stock_code, stock_data):
    """
    宏观因素对股票的影响分析
    """
    # region
    # 设置数据起始时间  df中时间数据都是升序排列
    start = stock_data.index[0]
    end = stock_data.index[-1]
    log.info(f"Start: {start}, End: {end}")
    # 获取宏观数据
    cpi_data = await MacroDataManage.get_macro_data(types=MacroDataEnum.CPI, start_date=str(start), end_date=str(end))
    ppi_data = await MacroDataManage.get_macro_data(types=MacroDataEnum.PPI, start_date=str(start), end_date=str(end))
    pmi_data = await MacroDataManage.get_macro_data(types=MacroDataEnum.PMI, start_date=str(start), end_date=str(end))
    # 格式化时间report_date并设置为设置索引
    cpi_data.set_index("report_date", inplace=True)
    ppi_data.set_index("report_date", inplace=True)
    pmi_data.set_index("report_date", inplace=True)
    # endregion
    #NOTE 数据库获取出的cpi(MOM) 和ppi(YOY)数据不用重新计算 pmi的则是原始数据
    cpi_data.columns = ["CPI_Inflation_Rate"]
    ppi_data.columns = ["PPI_Inflation_Rate"]
    pmi_data.columns = ["PMI"]
    # FIXME cpi ppi和股票的交互特征
    #合并两个数据（并集） 线性插值填数据
    macro_data = (pd.merge(left=cpi_data, right=ppi_data, left_index=True, right_index=True, how='outer')
                  .interpolate(method='linear'))
    #计算PMI(MOM)
    pmi_data["PMI_YOY"]=pmi_data["PMI"].pct_change().ffill()

    # 合并三个数据（并集） 线性插值填数据
    macro_data = (pd.merge(left=macro_data, right=pmi_data, left_index=True, right_index=True, how='outer')
                   .interpolate(method='linear'))
    macro_data.sort_index(inplace=True)
    # 将股票的收盘价数据和宏观数据合并
    stock_macro_data = stock_data[["close_price"]].join(macro_data, how='inner')

    # region
    # 画图 设置x轴时间跨度为5年
    # Closing Price
    plt.figure(figsize=(16, 10))
    plt.subplot(411)
    plt.plot(stock_macro_data.index, stock_macro_data['close_price'], label='Closing Price', color='blue')
    plt.title(f'{stock_code} Closing Price')
    plt.ylabel('Closing Price')
    plt.grid(True)
    plt.legend()
    # CPI_Inflation_Rate
    plt.subplot(412)
    plt.plot(stock_macro_data.index, stock_macro_data['CPI_Inflation_Rate'], label='CPI Inflation Rate', color='red')
    plt.title('CPI Inflation Rate (MOM)')

    plt.ylabel('Inflation Rate')
    plt.grid(True)
    plt.legend()
    # PPI_Inflation_Rate
    plt.subplot(413)
    plt.plot(stock_macro_data.index, stock_macro_data['PPI_Inflation_Rate'], label='PPI Inflation Rate', color='green')
    plt.title('PPI Inflation Rate (YOY)')
    plt.ylabel('Inflation Rate')
    plt.grid(True)
    plt.legend()
    # PMI_Ori
    plt.subplot(414)
    plt.plot(stock_macro_data.index, stock_macro_data['PMI_YOY'], label='PMI', color='purple')
    plt.title('PMI Change(MOM)')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('./picture/macro_correlation.svg')
    # 设置 X 轴时间格式
    for ax in plt.gcf().axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel('Date')
    plt.close()
    # endregion

    # NOTE 计算PMI 滚动窗口平均值与标准差(趋势和波动) (长期趋势与周期性分解) (PMI 与股票收益率的交互特征)
    # print(macro_data)
    # print("#" * 30)
    # print(cpi_data)
    # print("#" * 30)
    # print(ppi_data)
    # print("#" * 30)
    # print(pmi_data)

    # region
    # NOTE 提取三个宏观变量和股票收盘价的特征变量
    stock_macro_data.dropna(inplace=True)
    x=stock_macro_data[["CPI_Inflation_Rate", "PPI_Inflation_Rate", "PMI_YOY"]]
    y=stock_macro_data["close_price"]
    # NOTE 划分训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    # NOTE 构建线性回归模型
    # 初始化回归模型
    model = LinearRegression()
    # 训练模型
    model.fit(train_x, train_y)
    # 预测收盘价
    pred_train_y = model.predict(train_x)
    pred_test_y = model.predict(test_x)
    # 打印回归系数
    log.info(f'回归系数 (CPI_Inflation_Rate): {model.coef_[0]}')
    log.info(f'回归系数 (PPI_Inflation_Rate): {model.coef_[1]}')
    log.info(f'回归系数 (PMI_YOY): {model.coef_[2]}')
    log.info(f'截距 (intercept): {model.intercept_}')

    # 评估模型性能
    r2_train = r2_score(train_y, pred_train_y)
    r2_test = r2_score(test_y, pred_test_y)
    log.info(f'训练集 R²: {r2_train}')
    log.info(f'测试集 R²: {r2_test}')

    # 绘制实际值和预测值的对比
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(train_y, pred_train_y, color='blue', edgecolor='k', alpha=0.6)
    plt.plot([train_y.min(), train_y.max()], [train_y.min(), train_y.max()], color='red', linestyle='--')
    plt.title(f'Training set: Actual vs predicted values ({stock_code} closing prices)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    # 测试集实际值与预测值
    plt.subplot(2, 1, 2)
    plt.scatter(test_y, pred_test_y, color='green', edgecolor='k', alpha=0.6)
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], color='red', linestyle='--')
    plt.title(f'Test set: Actual vs predicted values ({stock_code} closing prices)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.show()
    plt.savefig('./picture/regression_actual_vs_pred.svg')
    plt.close()

    # 绘制三个宏观指标和股票收盘价的散点图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.scatter(stock_macro_data['CPI_Inflation_Rate'], stock_macro_data['close_price'], color='blue', alpha=0.6)
    plt.xlabel('CPI Inflation Rate(%)')
    plt.ylabel(f'{stock_code} Closing Prices')

    plt.subplot(1, 3, 2)
    plt.scatter(stock_macro_data['PPI_Inflation_Rate'], stock_macro_data['close_price'], color='green', alpha=0.6)
    plt.title('Relationship with Stock Closing Prices')
    plt.xlabel('PPI Inflation Rate(%)')
    plt.ylabel(f'{stock_code} Closing Prices')

    plt.subplot(1, 3, 3)
    plt.scatter(stock_macro_data['PMI_YOY'], stock_macro_data['close_price'], color='red', alpha=0.6)
    plt.xlabel('PMI YOY(%)')
    plt.ylabel(f'{stock_code} Closing Prices')
    plt.tight_layout()
    plt.show()
    plt.savefig('./picture/feature_and_close.svg')
    plt.close()
    # endregion

# 交易量和收盘价的关系分析
async def stock_volume_price_correlation(stock_code, stock_data):
    plt.figure(figsize=(10, 6))  # 设置图形大小
    #同时绘制散点图和线性回归线 regplot
    sns.regplot(x=stock_data['volume'].astype(int),
                y=stock_data['close_price'].astype(float),
                scatter_kws={'s': 10},
                line_kws={'color': 'red'})
    plt.title('Relationship Between Trading Volume and Close Price', fontsize=14)  # 图形标题
    plt.xlabel('Trading Volume', fontsize=12)  # x轴标签
    plt.ylabel('Close Price', fontsize=12)  # y轴标签
    plt.grid(True)  # 显示网格线
    plt.tight_layout()
    plt.show()
    plt.savefig('./picture/volume_and_close.svg')
    plt.close()
    # 计算交易量和收盘价之间的相关性
    volume = stock_data['volume'].astype(int)
    close_price = stock_data['close_price'].astype(float)
    correlation, p_value = pearsonr(volume, close_price)
    log.info(f"Pearson Correlation Coefficient: {correlation:.4f}")
    log.info(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        if correlation > 0:
            log.info("交易量和收盘价之间存在统计学上显著的正相关.")
        else:
            log.info("交易量和收盘价之间存在统计学上显著的负相关.")
    else:
        log.info("交易量和收盘价之间没有统计学上的显著相关性.")


async def main():
    await database.connect()
    await analyse_stock_data(stock_code=ExponentEnum.SZCZ.get_code(), start_date=None, end_date=None)
    await database.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
