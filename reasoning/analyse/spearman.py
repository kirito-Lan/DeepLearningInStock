import asyncio

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config.LoguruConfig import log
from constant.ExponentEnum import ExponentEnum
from manager.decoration.dbconnect import db_connection
from utils.ReFormatDate import format_date
from FeatureEngine import feature_engineering

@db_connection
async def hot_map(stock_code: str, start_date: str, end_date: str):
    start, end = format_date(start_date=start_date, end_date=end_date)
    # 读取数据
    df = pd.DataFrame()
    try:
        df = pd.read_csv(f'../processed_data/{stock_code}/feature_{stock_code}-{start}-{end}.csv')
        # 1. 剔除时间列
        df = df.drop('trade_date', axis=1)
    except Exception as e:
        log.info("文件不存在,进行特征工程构造")
        # 构造出来的不存在时间列，已经作为索引了
        await feature_engineering(stock_code, start, end)
        df = pd.read_csv(f'../processed_data/{stock_code}/feature_{stock_code}-{start}-{end}.csv')
        # 1. 剔除时间列
        df = df.drop('trade_date', axis=1)

    columns=["Open","High", "Low"]
    # 删除相关性高的
    for col in columns:
        del df[col]
    # 计算斯皮尔曼相关系数矩阵去df的前20列
    spearman_corr = df.iloc[:, :20].corr(method='spearman')
    #spearman_corr = df.corr(method='spearman')

    # 绘制热力图
    plt.figure(figsize=(15, 10))  # 调整为更紧凑的尺寸
    heatmap = sns.heatmap(
        spearman_corr,
        annot=True,
        fmt=".2f",  # 保留2位小数
        cmap='coolwarm',
        center=0,
        vmin=-1.0,  # 固定颜色范围
        vmax=1.0,
        linewidths=0.5,
        linecolor='black',
        cbar_kws={'shrink': 0.5}  # 调整颜色条大小
    )

    # 设置图表样式
    plt.title('Feature Correlation Matrix', fontsize=14, pad=15)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # 调整布局
    plt.tight_layout()
    plt.savefig(f'../picture/{stock_code}/spearman_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    asyncio.run(hot_map(ExponentEnum.HS300.get_code(), None, None))
