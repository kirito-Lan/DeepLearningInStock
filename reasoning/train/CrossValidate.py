import asyncio

import pandas as pd
import numpy as np
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from config.LoguruConfig import log
from constant.ExponentEnum import ExponentEnum
from manager.decoration.dbconnect import db_connection
from reasoning.analyse.FeatureEngine import feature_engineering
from utils.ReFormatDate import format_date

warnings.filterwarnings("ignore")

@db_connection
# 使用交叉验证构造时间序列数据集进行增强
async def build_model(stock_code: str, start_date: str, end_date: str):
    """构建并训练LSTM股票预测模型"""
    # 获取枚举值
    stock = ExponentEnum.get_enum_by_code(stock_code)
    if stock is None:
        raise Exception("股票代码不存在")
    # 格式化时间
    start_date, end_date = format_date(start_date=start_date, end_date=end_date)
    # 读取数据
    log.info(f"开始构建模型，读取股票代码：{stock_code} 数据")
    data = pd.DataFrame()
    try:
        data = pd.read_csv(f'../processed_data/{stock_code}/feature_{stock.get_code()}-{start_date}-{end_date}.csv')
    except FileNotFoundError:
        log.info(f"读取数据为空开始特征工程")
        await feature_engineering(stock_code,start_date,end_date)
        #特征工程结束后再次读取
        data = pd.read_csv(f'../processed_data/{stock_code}/feature_{stock.get_code()}-{start_date}-{end_date}.csv')

    if data is None or data.empty:
        raise Exception("没有找到对应的股票数据")
    data = data.rename(columns={'trade_date': 'Date'})
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    log.info(data.head())

    # 转换日期格式，并按照日期排序
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # 提取目标列
    target = 'Close'
    # 提取除目标列之外的所有特征列
    features = data.drop(columns=[target, 'Date','Open','High','Low']).values

    # 选取特征和目标
    X = features
    y = data[target].values.reshape(-1, 1)

    # 数据归一化 (MinMaxScaler)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)


    # 创建输入序列和标签，使用滑动窗口
    def create_sequences(X, y, time_steps=60):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:i + time_steps])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)


    # 2. 数据增强与划分数据集
    def moving_average_smoothing(data, window_size=3):
        """对多维时间序列数据中的每个特征分别进行平滑处理"""
        smoothed_data = np.empty_like(data)  # 创建与输入数据相同维度的空数组
        for col in range(data.shape[1]):  # 对每一列（每个特征）进行平滑处理
            smoothed_data[:, col] = np.convolve(data[:, col], np.ones(window_size) / window_size, mode='same')
        return smoothed_data

    def data_augmentation(X, y, num_augmentations=5):
        """进行时间序列数据增强"""
        X_augmented = []
        y_augmented = []

        for i in range(len(X)):
            # 原始数据
            X_augmented.append(X[i])
            y_augmented.append(y[i])

            # 增强方法1: 平滑处理
            X_smooth = moving_average_smoothing(X[i])
            # 增强方法2: 添加噪声
            X_noisy = X_smooth + 0.01 * np.random.randn(*X_smooth.shape)
            # 增强方法3: 时间偏移
            X_shifted = np.roll(X_noisy, np.random.randint(-5, 5), axis=0)

            # 保存增强数据
            X_augmented.append(X_shifted)
            y_augmented.append(y[i])  # 标签不变

        return np.array(X_augmented), np.array(y_augmented)


    #3. 测试不同滑动窗口大小的效果
    window_sizes = [ 20, 60, 90]  # 不同的滑动窗口大小
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']

    # 存储不同窗口大小下的结果
    results = {window_size: {metric: [] for metric in metrics} for window_size in window_sizes}

    for window_size in window_sizes:
        print(f'Processing window size: {window_size}')

        # 创建不同窗口大小的输入序列
        X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, window_size)

        # 将原始数据划分为训练集和测试集，保留测试集用于最终评估
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

        # 对训练集进行数据增强
        X_train_full_augmented, y_train_full_augmented = data_augmentation(X_train_full, y_train_full)

        # 使用 TimeSeriesSplit 进行交叉验证
        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_index, val_index) in enumerate(tscv.split(X_train_full_augmented)):
            print(f"Fold {fold + 1} for window size {window_size}")

            # 划分训练集和验证集
            X_train, X_val = X_train_full_augmented[train_index], X_train_full_augmented[val_index]
            y_train, y_val = y_train_full_augmented[train_index], y_train_full_augmented[val_index]

            # 模型架构
            model = Sequential()
            model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
            model.add(LSTM(64, return_sequences=True, kernel_regularizer=L2(0.001)))
            model.add(Dropout(0.3))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.3))
            # 输出层
            model.add(Dense(1))

            # 编译模型
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

            # 训练模型
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                                callbacks=[early_stopping], verbose=1)

            # 在验证集上进行预测
            val_predictions = model.predict(X_val)
            val_predicted_prices = scaler_y.inverse_transform(val_predictions)
            val_real_prices = scaler_y.inverse_transform(y_val.reshape(-1, 1))

            # 计算评价指标
            val_mse = mean_squared_error(val_real_prices, val_predicted_prices)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(val_real_prices, val_predicted_prices)
            val_r2 = r2_score(val_real_prices, val_predicted_prices)

            # 记录当前窗口大小和 fold 的指标
            results[window_size]['MSE'].append(val_mse)
            results[window_size]['RMSE'].append(val_rmse)
            results[window_size]['MAE'].append(val_mae)
            results[window_size]['R2'].append(val_r2)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'../results/window_size_performance_{stock_code}.csv', index=False)

    # 4. 计算每个窗口大小下的指标平均值，并绘制条形图
    averages = {window_size: {metric: np.mean(results[window_size][metric]) for metric in metrics} for window_size in window_sizes}

    # 对指标进行对数变换（仅对 MSE 和 RMSE 进行处理）
    log_transformed_results = {window_size: {
        'MSE': np.log1p(averages[window_size]['MSE']),   # 对 MSE 进行对数变换
        'RMSE': np.log1p(averages[window_size]['RMSE']),  # 对 RMSE 进行对数变换
        'MAE': np.log1p(averages[window_size]['MAE']),  # 对 MAE 进行对数变换
        'R2': averages[window_size]['R2']     # 保持 R² 不变
    } for window_size in window_sizes}

    # 设置条形图
    # 调整 barWidth 和 xticks 间距
    barWidth = 0.15  # 增大条形图的宽度
    r = np.arange(len(metrics))  # 设置基础横坐标位置
    plt.figure(figsize=(12, 6))  # 调整图形大小

    # 使用Seaborn调色板
    colors = sns.color_palette("Blues", len(window_sizes))  # 使用更加美观的调色板

    # 绘制不同窗口大小的指标
    for idx, window_size in enumerate(window_sizes):
        avg_metrics = [log_transformed_results[window_size][metric] for metric in metrics]
        bars = plt.bar(r + idx * barWidth, avg_metrics, width=barWidth, color=colors[idx], label=f'Window Size {window_size}')

        # 在每个条形图上显示数值
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    # 添加图例和标签
    plt.xlabel('Metrics', fontweight='bold')
    plt.ylabel('Log-Transformed / Original Value', fontweight='bold')

    # 调整 xticks 的位置，使其位于每组条形图的中央
    plt.xticks([r + barWidth * (len(window_sizes) / 2 - 0.5) for r in np.arange(len(metrics))], metrics)

    plt.title('Log-Transformed Evaluation Metrics for Different Window Sizes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../picture/{stock_code}/window_size_performance.png')



if __name__ == '__main__':

    asyncio.run(build_model(ExponentEnum.CYB.get_code(),None,None))
