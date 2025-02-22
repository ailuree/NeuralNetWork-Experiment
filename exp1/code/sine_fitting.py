import torch                                 # 导入torch库
import torch.nn as nn                        # torch.nn 是pytorch的神经网络库
import torch.optim as optim                  # torch.optim 是pytorch的优化库
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
torch.manual_seed(42)                       
np.random.seed(42)

# 生成带有白噪声的正弦数据  默认100个点
def generate_noisy_sine_data(num_points=200):
    x = np.linspace(0, 2*np.pi, num_points)  # 在0到2π之间生成均匀分布的点
    y = np.sin(x) + np.random.normal(0, 0.1, num_points)  # 在正弦函数上添加均值为0，标准差为0.1的高斯噪声
    return x, y

# 定义BP神经网络模型
class BPNeuralNetwork(nn.Module):
    # 初始化 定义网络结构
    def __init__(self):
        # Linear 是线性变换，作用是进行线性变换，将输入的特征按照权重进行线性变换，然后加上偏置再映射到输出空间
        super(BPNeuralNetwork, self).__init__()
        # 输入层认为是一个神经元，输入一个特征
        self.hidden1 = nn.Linear(1, 20)   # 隐藏层1，输入1个特征，输出20个特征    
        self.hidden2 = nn.Linear(20, 20)  # 隐藏层2，输入20个特征，输出20个特征
        self.hidden3 = nn.Linear(20, 10)  # 隐藏层3，输入20个特征，输出10个特征
        self.output = nn.Linear(10, 1)    # 输出层，输入10个特征，输出1个特征        
    # 前向传播
    def forward(self, x):
        # 使用ReLU激活函数  ReLU函数的表达式  为：f(x) = max(0, x)  
        x = torch.relu(self.hidden1(x))   # 输入数据通过第一个隐藏层进行线性变换，ReLU激活
        x = torch.relu(self.hidden2(x))   # 第一个隐藏层的输出通过第二个隐藏层进行线性变换，ReLU激活
        x = torch.relu(self.hidden3(x))   # 第二个隐藏层的输出通过第三个隐藏层进行线性变换，ReLU激活
        x = self.output(x)                # 第三个隐藏层的输出通过输出层进行线性变换之后输出最终结果
        return x

# 训练模型 默认设定训练epoch为2000，学习率为0.005
def train_model(model, x, y, num_epochs=2000, learning_rate=0.005):  
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数   MSE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器   Adam优化器是一种自适应学习率的优化算法 
    
    for epoch in range(num_epochs):
        inputs = torch.FloatTensor(x).view(-1, 1)  # 将输入转换为PyTorch张量
        targets = torch.FloatTensor(y).view(-1, 1) 
        
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数   这里的更新参数其实就是更新权重和偏置
        
        # 每隔100个epoch打印一次损失
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 计算均方误差（MSE）和均方根误差（RMSE）
def evaluate_model(model, x, y):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        inputs = torch.FloatTensor(x).view(-1, 1)
        targets = torch.FloatTensor(y).view(-1, 1)
        outputs = model(inputs)
        mse = nn.MSELoss()(outputs, targets).item()
        rmse = np.sqrt(mse)
    return mse, rmse

# 计算平均误差
def calculate_mean_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# 主函数
def main():
    # 设置matplotlib使用系统默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 生成数据  带有白噪声的正弦数据
    x, y = generate_noisy_sine_data()
    
    # 创建和训练模型
    model = BPNeuralNetwork()
    train_model(model, x, y)
    
    # 使用训练好的模型进行预测
    with torch.no_grad():
        # 生成测试数据  np.linspace(0, 2*np.pi, 100) 在0到2π之间生成100个均匀分布的点
        x_test = torch.FloatTensor(np.linspace(0, 2*np.pi, 100)).view(-1, 1)
        # 使用训练好的模型对每个输入数据进行预测
        y_pred = model(x_test).numpy().flatten()
    
    # 评估模型
    mse, rmse = evaluate_model(model, x, y)
    print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}')
    
    # 计算标准正弦函数的值
    y_true = np.sin(np.linspace(0, 2*np.pi, 100))
    
    # 计算平均误差
    mean_error = calculate_mean_error(y_true, y_pred)
    print(f'平均误差: {mean_error:.4f}')
    
    # 绘制结果 
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='带噪声的样本数据')
    plt.plot(np.linspace(0, 2*np.pi, 100), np.sin(np.linspace(0, 2*np.pi, 100)), 'r', label='标准正弦曲线')
    plt.plot(x_test, y_pred, 'g', label='BP神经网络拟合曲线')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('BP神经网络拟合正弦函数')
    plt.show()

if __name__ == "__main__":
    main()