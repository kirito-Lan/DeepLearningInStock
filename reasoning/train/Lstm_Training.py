import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from config.LoguruConfig import log
from constant.ExponentEnum import ExponentEnum


def build_model(stock_code: str):
    # 获取枚举值
    stock = ExponentEnum.get_enum_by_code(stock_code)
    if stock is None:
        raise Exception("股票代码不存在")

    # 读取数据
    log.info(f"开始构建模型，读取股票代码：{stock_code} 数据")
    data = pd.read_csv(f'../processed_data/feature_{stock.get_code()}.csv')
    # 转换日期格式，并按照日期排序
    data = data.rename(columns={'trade_date': 'Date'})
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    log.info(data.head())
    # 提取目标列
    target = 'Close'
    # 提取除目标列之外的所有特征列
    features = data.drop(columns=[target, 'Date']).values
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

    time_steps = 60
    X_lstm, y_lstm = create_sequences(X_scaled, y_scaled, time_steps)

    # 划分数据集
    log.info('划分数据集...')
    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X_lstm, y_lstm, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    # 构建LSTM模型
    log.info('构建LSTM模型...')
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()

    log.info('训练LSTM模型...')
    # 训练模型时，使用验证集评估
    history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val))

    # 预测（使用保留的测试集进行预测）
    log.info('预测LSTM模型...')
    predictions = model.predict(X_test)

    # 反归一化
    predicted_prices = scaler_y.inverse_transform(predictions)
    real_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # 计算评价指标
    mse = mean_squared_error(real_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)

    # 输出各个指标
    log.info(f"Mean Squared Error (MSE): {mse}")
    log.info(f"Root Mean Squared Error (RMSE): {rmse}")
    log.info(f"Mean Absolute Error (MAE): {mae}")
    log.info(f"R² Score: {r2}")

    plt.figure(figsize=(8, 4))

    # 绘制真实股价
    plt.plot(data['Date'][-len(real_prices):], real_prices, label="Real Prices", color='blue')

    # 绘制预测股价
    plt.plot(data['Date'][-len(predicted_prices):], predicted_prices, label="Predicted Prices", color='red',
             linestyle='--')

    plt.title('Real vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../picture/stock_prices.svg')
    plt.show()
    plt.close()
    # 绘制训练损失和验证损失的变化
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    plt.savefig('../picture/loss.svg')
    plt.show()
    plt.close()

    # 使用 Pandas 保存预测结果到 CSV 文件
    predicted_df = pd.DataFrame(predicted_prices, columns=["Predicted Close"])
    predicted_df.to_csv(f'../results/predicted_stock_prices_{stock.get_code()}.csv', index=False)

if __name__ == '__main__':
    build_model('399001')
