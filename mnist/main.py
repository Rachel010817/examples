#!/anaconda/bin/python3 #python解释器位置
from __future__ import print_function#和将来版本兼容
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #优化器
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR #LR是learning rate


class Net(nn.Module):   #定义一个网络Net，网络由2层的卷积构成
    def __init__(self):     #定义网络有什么层
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)    # 卷积层 输入通道数为1，输出通道数为32
        self.conv2 = nn.Conv2d(32, 64, 3, 1)    # 输入通道数为32，输出通道数为64.    Applies a 2D convolution over an input signal composed of several input planes.
        self.dropout1 = nn.Dropout(0.25)    #让某个神经元的激活值以一定的概率p（默认是0.5）停止工作，这样可以使模型泛化性更强，防止过拟合
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128) #Applies a linear transformation to the incoming data: y = xA^T + b
        self.fc2 = nn.Linear(128, 10)   #两个全连接层

    def forward(self, x):   #定义网络的前向传播的过程
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)#Applies a 2D max pooling over an input signal composed of several input planes.
        x = self.dropout1(x)
        x = torch.flatten(x, 1) #Flattens a contiguous range of dims in a tensor.拉平降维
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)    #Softmax的含义就在于不再唯一的确定某一个最大值，而是为每个输出分类的结果都赋予一个概率值，表示属于每个类别的可能性。
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()   #开始训练。model在
    for batch_idx, (data, target) in enumerate(train_loader):#
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()#所有参数清零
        output = model(data)#把参数算一遍
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()    #梯度下降
        if batch_idx % args.log_interval == 0:  #检查一下log_interval记的文件
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())) #train_loader迭代器，里面有很多数据样本
            if args.dry_run:    #如果dry_run
                break

#训练完了，测试
def test(model, device, test_loader):
    model.eval()#eval从nn.model继承的，包含在开头 super(Net, self).__init__()里面的函数
    test_loss = 0
    correct = 0
    with torch.no_grad():#测试时不需要计算导数，torch model里面会自动算，把它关上
        for data, target in test_loader:#载入数据
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability返回指定维度最大值的序号
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))#看看正确的多少

#主程序
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')#定义单个的命令行参数应当如何解析.自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',  #规定批尺寸
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',  #训练14次
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args() #命令行里输入的内容（没有就用上面的default）
    use_cuda = not args.no_cuda and torch.cuda.is_available() #有没有可用的CUDA（GPU）

    torch.manual_seed(args.seed)    #Sets the seed for generating random numbers

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}#数据载入
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([  #输入图像转化 标准化参数
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, 
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)#训练集
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)#测试集

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)#可以输入学习率
    #model.parameters来自nn.model

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)#把optimizer加上学习率衰减的功能。StepLR在库里，LR：learning rate
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)#训练
        test(model, device, test_loader)#测试
        scheduler.step()    #学习率

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")#保存训练好的模型参数，下回直接能用了


if __name__ == '__main__':#别的程序直接import这个文件不运行，直接运行这个（名字的）程序才能用，不然import一次运行一次
    main()#调用主程序
    
    
