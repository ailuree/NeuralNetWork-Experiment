import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('highway_transport_data.csv', encoding='gbk')

# 打印数据信息，查看是否有缺失值
print(data.info())

# 处理缺失值
data = data.dropna()  # 删除包含NaN的行

# 准备输入特征和目标变量
X = data[['人数(万人)', '机动车数(万辆)', '公路面积(万平方公里)']].values
y_passengers = data['公路客运量(万人)'].values
y_freight = data['公路货运量(万吨)'].values

# 数据归一化
scaler_X = MinMaxScaler()
scaler_y_passengers = MinMaxScaler()
scaler_y_freight = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_passengers_scaled = scaler_y_passengers.fit_transform(y_passengers.reshape(-1, 1))
y_freight_scaled = scaler_y_freight.fit_transform(y_freight.reshape(-1, 1))

# 创建并训练BP神经网络模型
model_passengers = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
model_freight = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)

model_passengers.fit(X_scaled, y_passengers_scaled.ravel())
model_freight.fit(X_scaled, y_freight_scaled.ravel())

# 设置matplotlib使用系统默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 绘制拟合曲线
plt.figure(figsize=(12, 5))

# 客运量拟合曲线
plt.subplot(1, 2, 1)
plt.plot(data['年份'], y_passengers, 'bo-', label='实际客运量')
predicted_passengers = scaler_y_passengers.inverse_transform(model_passengers.predict(X_scaled).reshape(-1, 1))
plt.plot(data['年份'], predicted_passengers, 'r-', label='拟合客运量')
plt.title('公路客运量拟合曲线')
plt.xlabel('年份')
plt.ylabel('公路客运量(万人)')
plt.legend()

# 货运量拟合曲线
plt.subplot(1, 2, 2)
plt.plot(data['年份'], y_freight, 'bo-', label='实际货运量')
predicted_freight = scaler_y_freight.inverse_transform(model_freight.predict(X_scaled).reshape(-1, 1))
plt.plot(data['年份'], predicted_freight, 'r-', label='拟合货运量')
plt.title('公路货运量拟合曲线')
plt.xlabel('年份')
plt.ylabel('公路货运量(万吨)')
plt.legend()

plt.tight_layout()
plt.show()

# 预测2010年和2011年的客运量和货运量
prediction_data = np.array([[73.39, 3.9, 0.98], [75.55, 4.1, 1.02]])
prediction_data_scaled = scaler_X.transform(prediction_data)

predicted_passengers = scaler_y_passengers.inverse_transform(model_passengers.predict(prediction_data_scaled).reshape(-1, 1))
predicted_freight = scaler_y_freight.inverse_transform(model_freight.predict(prediction_data_scaled).reshape(-1, 1))

print("预测结果：")
print("2010年预测公路客运量：{:.2f} 万人".format(predicted_passengers[0][0]))
print("2010年预测公路货运量：{:.2f} 万吨".format(predicted_freight[0][0]))
print("2011年预测公路客运量：{:.2f} 万人".format(predicted_passengers[1][0]))
print("2011年预测公路货运量：{:.2f} 万吨".format(predicted_freight[1][0]))