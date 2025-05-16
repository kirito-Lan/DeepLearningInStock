import asyncio
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import L2

from config.LoguruConfig import log, project_root
from constant.ExponentEnum import ExponentEnum
from manager.decoration.dbconnect import db_connection
import seaborn as sns
from reasoning.analyse.FeatureEngine import feature_engineering
from utils.ReFormatDate import format_date

# 固定随机种子保证训练结果可以复现
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


file_root= str(project_root/"reasoning/")

async def build_model(stock_code: str, start_date: str, end_date: str, epochs:int,reg:float,dropout:float):
    """构建并训练LSTM股票预测模型"""
    log.info(f"开始构建模型epochs: 【{epochs}】reg: 【{reg}】dropout: 【{dropout}】")
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

    # 数据预处理
    target = 'Close'
    features = data.drop(columns=[target, 'Date'])
    X = features.values
    y = data[target].values.reshape(-1, 1)

    # 归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 模型训练配置
    window_lengths = [20, 30, 60]
    best_metrics = {
        'window': None,
        'model': None,
        'val_loss': float('inf'),
        'history': None
    }
    temp_files = []

    # 开始窗口长度搜索
    for time_steps in window_lengths:
        log.info(f"正在训练 {time_steps} 天窗口模型...")

        # 创建时间序列数据
        def create_sequences(X, y, steps):
            Xs, ys = [], []
            for i in range(len(X) - steps):
                Xs.append(X[i:i + steps])
                ys.append(y[i + steps])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

        # 数据集划分（同时保持时序）
        X_train, X_rem, y_train, y_rem = train_test_split(X_seq, y_seq, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, shuffle=False)
        log.debug(f"训练集shape：{X_train.shape}, 验证集shape：{X_val.shape}, 测试集shape：{X_test.shape}")
        # 模型架构
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        # 添加L2正则化减少过拟合
        model.add(LSTM(64, return_sequences=True, kernel_regularizer=L2(reg)))
        model.add(Dropout(dropout))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(dropout))
        # 输出层
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
        # 输出模型摘要信息
        model.summary()
        # 回调设置
        checkpoint_path = f'../model/{stock.get_code()}_temp_{time_steps}.keras'
        temp_files.append(checkpoint_path)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, save_best_only=True)
        ]

        # 训练过程
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # 记录最佳模型
        current_best_loss = min(history.history['val_loss'])
        if current_best_loss < best_metrics['val_loss']:
            best_metrics.update({
                'window': time_steps,
                'val_loss': current_best_loss,
                'history': history.history,
                'model': tf.keras.models.load_model(checkpoint_path)
            })

    # 最佳模型处理
    if not best_metrics['model']:
        raise RuntimeError("没有找到有效模型")

    log.info(f"最佳窗口长度: {best_metrics['window']} 天，验证损失: {best_metrics['val_loss']:.4f}")

    # 清理临时文件
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
            log.debug(f"已删除临时文件: {file}")

    # 最终模型保存
    final_model_path = f'../model/{stock.get_name()}_model.keras'
    best_metrics['model'].save(final_model_path)
    log.info(f"模型已保存至: {final_model_path}")

    # 使用最佳窗口重新生成测试集
    def create_best_sequences(X, y):
        return create_sequences(X, y, best_metrics['window'])

    X_seq_full, y_seq_full = create_best_sequences(X_scaled, y_scaled)
    _, X_test_final, _, y_test_final = train_test_split(
        X_seq_full, y_seq_full,
        test_size=0.15,
        shuffle=False
    )

    # 预测与评估
    predictions = best_metrics['model'].predict(X_test_final)
    pred_prices = scaler_y.inverse_transform(predictions)
    true_prices = scaler_y.inverse_transform(y_test_final)

    # 计算指标
    mse = mean_squared_error(true_prices, pred_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_prices, pred_prices)
    r2 = r2_score(true_prices, pred_prices)

    # 打印模型评估结果
    log.info(f"""
    模型评估结果:
    MSE: {mse:.2f}
    RMSE: {rmse:.2f}
    MAE: {mae:.2f}
    R²: {r2:.2f}
    """)

    # 模型评估指标可视化
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    original_values = [mse, rmse, mae, r2]

    # 对所有指标进行对数变换（除了 R²）
    log_values = [np.log1p(v) if v > 0 else v for v in original_values[:-1]]
    log_values.append(original_values[-1])  # R² 保持不变

    plt.figure(figsize=(10, 6))

    # 使用Seaborn调色板
    colors = sns.color_palette("Blues", len(metrics))

    # 绘制条形图
    plt.bar(metrics, log_values, color=colors)
    plt.yscale('log')  # 设置纵坐标为对数坐标
    plt.title('Model Evaluation Metrics (Log Scale)')
    plt.ylabel('Value (log scale)')
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # 在柱子上方添加数值标签
    for i, v in enumerate(log_values):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')


    plt.savefig(f'../picture/{stock_code}/evaluation_metrics.png')
    plt.close()

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'].values[-len(true_prices):], true_prices, label='Actual')
    plt.plot(data['Date'].values[-len(pred_prices):], pred_prices, label='Predicted', linestyle='--')
    plt.title(f"{stock.get_code()} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f'../picture/{stock_code}/prediction_result.png')
    plt.close()

    # 训练过程可视化
    plt.figure(figsize=(10, 5))
    plt.plot(best_metrics['history']['loss'], label='Training Loss')
    plt.plot(best_metrics['history']['val_loss'], label='Validation Loss')
    plt.title(f"Training Process (Best Window: {best_metrics['window']} days)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'../picture/{stock_code}/training_history.png')
    plt.close()

    # 保存预测结果
    result_df = pd.DataFrame({
        'Date': data['Date'].values[-len(pred_prices):],
        'Actual': true_prices.flatten(),
        'Predicted': pred_prices.flatten()
    })
    result_df.to_csv(f'{file_root}/results/predicted_close{stock.get_code()}.csv', index=False)

    # 保存训练损失和验证损失
    pd.DataFrame({
        'Epochs': list(range(1, len(best_metrics['history']['loss']) + 1)),
        'Training Loss': best_metrics['history']['loss'],
        'Validation Loss': best_metrics['history']['val_loss']
    }).to_csv(f'{file_root}/results/training_losses_{stock.get_code()}.csv', index=False)

    return True


@db_connection
async def main():
    await build_model(stock_code=ExponentEnum.SZCZ.get_code(),start_date=None,end_date=None,epochs=150, reg=0.001 ,dropout=0.3)


if __name__ == "__main__":
    asyncio.run(main())