import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

# 锚定到脚本所在目录，确保路径不依赖工作目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

# 1. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")

# 2. 数据预处理与增强
# 训练集：归一化 + 数据增强
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),               # 随机旋转 ±10°
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移 ±10%
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 测试集：仅归一化（不做增强）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
torchvision.datasets.MNIST.resources = [
    ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", None),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", None),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", None),
    ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", None),
]

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. 定义CNN模型（增强版）
class ImprovedCNN(nn.Module):
    """改进的CNN模型：添加BatchNorm + Dropout + 更深的卷积层"""
    def __init__(self, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()
        # 卷积块1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)                    # 批归一化：加速收敛、稳定训练
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)                  # 28→14
        self.dropout1 = nn.Dropout2d(0.25)               # 空间Dropout：防止过拟合

        # 卷积块2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)                  # 14→7
        self.dropout2 = nn.Dropout2d(0.25)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 卷积块1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # 卷积块2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # 全连接
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

model = ImprovedCNN(dropout_rate=0.5).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 4. 损失函数、优化器、学习率调度、TensorBoard
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器：当验证损失停滞时降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# 初始化TensorBoard，日志存在./runs/mnist_cnn_improved里
writer = SummaryWriter(log_dir='./runs/mnist_cnn_improved')

# 把模型结构写入TensorBoard（用一批数据做示例）
example_data, example_targets = next(iter(train_loader))
example_data = example_data.to(device)
writer.add_graph(model, example_data)

# 5. 训练循环
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 每100个batch记录一次训练损失
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
            writer.add_scalar('Train/Loss_step', loss.item(), epoch * len(train_loader) + batch_idx)
    
    # 记录每个epoch的平均训练损失
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} 平均训练损失: {avg_train_loss:.4f}")
    writer.add_scalar('Train/Loss_epoch', avg_train_loss, epoch)
    return avg_train_loss

# 6. 测试循环
def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\n测试集: 平均损失: {avg_test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    
    # 记录测试损失和准确率
    writer.add_scalar('Test/Loss', avg_test_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)
    return avg_test_loss, accuracy

# 7. 开始训练
if __name__ == "__main__":
    epochs = 10
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\n")
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train(model, device, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion, epoch)
        
        # 学习率调度：根据测试损失调整学习率
        scheduler.step(test_loss)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        print(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './best_mnist_cnn.pth')
            print(f"最佳模型已保存！准确率: {best_acc:.2f}%")
    
    # 关闭TensorBoard写入器
    writer.close()
    print(f"\n")
    print(f"训练完成！最佳测试准确率: {best_acc:.2f}%")
    print(f"日志已保存到 ./runs/mnist_cnn_improved")
    print(f"最佳模型已保存到 ./best_mnist_cnn.pth")