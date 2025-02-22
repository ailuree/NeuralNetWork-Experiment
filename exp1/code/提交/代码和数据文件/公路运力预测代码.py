import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 定义优化后的神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 16)    # 隐藏层1 输入3 输出16
        self.fc2 = nn.Linear(16, 8)    # 隐藏层2 输入16 输出8
        self.fc3 = nn.Linear(8, 1)     # 输出层 输入8 输出1
        self.relu = nn.ReLU()          # 激活函数
        self.dropout = nn.Dropout(0.1)  # 添加Dropout层

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义评估函数
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # 计算归一化误差
    y_mean = np.mean(y_true)
    normalized_mae = mae / y_mean * 100  # 转换为百分比
    normalized_mse = mse / (y_mean ** 2) * 100  # 转换为百分比
    normalized_rmse = rmse / y_mean * 100  # 转换为百分比
    
    return mae, mse, rmse, r2, normalized_mae, normalized_mse, normalized_rmse

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

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X_scaled)
y_passengers_tensor = torch.FloatTensor(y_passengers_scaled)
y_freight_tensor = torch.FloatTensor(y_freight_scaled)

# 创建模型实例
model_passengers = Net()
model_freight = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer_passengers = optim.Adam(model_passengers.parameters(), lr=0.001, weight_decay=1e-5)
optimizer_freight = optim.Adam(model_freight.parameters(), lr=0.001, weight_decay=1e-5)

# 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 客运量模型
    outputs_passengers = model_passengers(X_tensor)
    loss_passengers = criterion(outputs_passengers, y_passengers_tensor)
    optimizer_passengers.zero_grad()
    loss_passengers.backward()
    optimizer_passengers.step()

    # 货运量模型
    outputs_freight = model_freight(X_tensor)
    loss_freight = criterion(outputs_freight, y_freight_tensor)
    optimizer_freight.zero_grad()
    loss_freight.backward()
    optimizer_freight.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss Passengers: {loss_passengers.item():.4f}, Loss Freight: {loss_freight.item():.4f}')

# 在训练数据上进行预测
model_passengers.eval()
model_freight.eval()
with torch.no_grad():
    y_passengers_pred = model_passengers(X_tensor).numpy()
    y_freight_pred = model_freight(X_tensor).numpy()

# 将预测结果转换回原始尺度
y_passengers_pred = scaler_y_passengers.inverse_transform(y_passengers_pred)
y_freight_pred = scaler_y_freight.inverse_transform(y_freight_pred)

# 评估模型
mae_passengers, mse_passengers, rmse_passengers, r2_passengers, norm_mae_passengers, norm_mse_passengers, norm_rmse_passengers = evaluate_model(y_passengers, y_passengers_pred)
mae_freight, mse_freight, rmse_freight, r2_freight, norm_mae_freight, norm_mse_freight, norm_rmse_freight = evaluate_model(y_freight, y_freight_pred)

print("\n客运量模型评估:")
print(f"平均绝对误差 (MAE): {mae_passengers:.2f} 万人 (归一化: {norm_mae_passengers:.2f}%)")
print(f"均方误差 (MSE): {mse_passengers:.2f} (归一化: {norm_mse_passengers:.2f}%)")
print(f"均方根误差 (RMSE): {rmse_passengers:.2f} 万人 (归一化: {norm_rmse_passengers:.2f}%)")
print(f"决定系数 (R²): {r2_passengers:.4f}")

print("\n货运量模型评估:")
print(f"平均绝对误差 (MAE): {mae_freight:.2f} 万吨 (归一化: {norm_mae_freight:.2f}%)")
print(f"均方误差 (MSE): {mse_freight:.2f} (归一化: {norm_mse_freight:.2f}%)")
print(f"均方根误差 (RMSE): {rmse_freight:.2f} 万吨 (归一化: {norm_rmse_freight:.2f}%)")
print(f"决定系数 (R²): {r2_freight:.4f}")

# 设置matplotlib使用系统默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 绘制拟合曲线
plt.figure(figsize=(12, 5))

# 客运量拟合曲线
plt.subplot(1, 2, 1)
plt.plot(data['年份'], y_passengers, 'bo-', label='实际客运量')
plt.plot(data['年份'], y_passengers_pred, 'r-', label='拟合客运量')
plt.title('公路客运量拟合曲线')
plt.xlabel('年份')
plt.ylabel('公路客运量(万人)')
plt.legend()

# 货运量拟合曲线
plt.subplot(1, 2, 2)
plt.plot(data['年份'], y_freight, 'bo-', label='实际货运量')
plt.plot(data['年份'], y_freight_pred, 'r-', label='拟合货运量')
plt.title('公路货运量拟合曲线')
plt.xlabel('年份')
plt.ylabel('公路货运量(万吨)')
plt.legend()

plt.tight_layout()
plt.show()

# 添加误差分析图
plt.figure(figsize=(12, 5))

# 客运量误差分析
plt.subplot(1, 2, 1)
plt.scatter(y_passengers, y_passengers_pred)
plt.plot([y_passengers.min(), y_passengers.max()], [y_passengers.min(), y_passengers.max()], 'r--', lw=2)
plt.xlabel('实际客运量')
plt.ylabel('预测客运量')
plt.title('客运量预测误差分析')

# 货运量误差分析
plt.subplot(1, 2, 2)
plt.scatter(y_freight, y_freight_pred)
plt.plot([y_freight.min(), y_freight.max()], [y_freight.min(), y_freight.max()], 'r--', lw=2)
plt.xlabel('实际货运量')
plt.ylabel('预测货运量')
plt.title('货运量预测误差分析')

plt.tight_layout()
plt.show()

# 预测2010年和2011年的客运量和货运量
prediction_data = np.array([[73.39, 3.9, 0.98], [75.55, 4.1, 1.02]])
prediction_data_scaled = scaler_X.transform(prediction_data)
prediction_tensor = torch.FloatTensor(prediction_data_scaled)

with torch.no_grad():
    predicted_passengers = scaler_y_passengers.inverse_transform(model_passengers(prediction_tensor).numpy())
    predicted_freight = scaler_y_freight.inverse_transform(model_freight(prediction_tensor).numpy())

print("预测结果：")
print("2010年预测公路客运量：{:.2f} 万人".format(predicted_passengers[0][0]))
print("2010年预测公路货运量：{:.2f} 万吨".format(predicted_freight[0][0]))
print("2011年预测公路客运量：{:.2f} 万人".format(predicted_passengers[1][0]))
print("2011年预测公路货运量：{:.2f} 万吨".format(predicted_freight[1][0]))