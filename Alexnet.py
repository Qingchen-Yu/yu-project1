import torch
import torchvision.models as models
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH=2
BATCH_SIZE=256


transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                       train=True,
                                       transform=transform,
                                       target_transform=None,
                                       download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar',
                                       train=False,
                                       transform=transform,
                                       target_transform=None,
                                       download=True)
                                       
train_loader = DataLoader(dataset=train_dataset, # 传入的数据集, 必须参数
                               batch_size=32,       # 输出的batch大小
                               shuffle=True,       # 数据是否打乱
                               num_workers=2)      # 进程数, 0表示只有主进程
test_loader = DataLoader(dataset=test_dataset, # 传入的数据集, 必须参数
                               batch_size=32,       # 输出的batch大小
                               shuffle=False,       # 数据是否打乱
                               num_workers=2)      # 进程数, 0表示只有主进程



#数据形状[32, 3, 32, 32]




class AlexNet(nn.Module):

    def __init__(self,  num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

#定义模型


model=AlexNet().to(device)

#优化器的选择
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)

def train_model(model,device,train_loader,optimizer,epoch):
    train_loss=0
    model.train()
    
    for batch_index,(data,label) in enumerate(train_loader):
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,label)
        loss.backward()
        optimizer.step()
        if batch_index%300==0:
            train_loss=loss.item()
            print('Train Epoch:{}\ttrain loss:{:.6f}'.format(epoch,loss.item()))
 
    return  train_loss

def test_model(model,device,test_loader):
    model.eval()
    correct=0.0
    test_loss=0.0
    
        #不需要梯度的记录
    with torch.no_grad():
        for data,label in test_loader:
            data,label=data.to(device),label.to(device)
            output=model(data)
            test_loss+=F.cross_entropy(output,label).item()
            pred=output.argmax(dim=1)
            correct+=pred.eq(label.view_as(pred)).sum().item()
        test_loss/=len(test_loader.dataset)
        print('Test_average_loss:{:.4f},Accuracy:{:3f}\n'.format(
            test_loss,100*correct/len(test_loader.dataset)
        ))
        acc=100*correct/len(test_loader.dataset)
    
        return test_loss,acc

list=[]
Train_Loss_list=[]
Valid_Loss_list=[]
Valid_Accuracy_list=[]
 
#Epoc的调用
for epoch in range(1,EPOCH+1):
    #训练集训练
    train_loss=train_model(model,device,train_loader,optimizer,epoch)
    Train_Loss_list.append(train_loss)
    #torch.save(model,r'E:\buttle\kaggle\save_model\model%s.pth'%epoch)
 
    #验证集进行验证
    test_loss,acc=test_model(model,DEVICE,valid_loader)
    Valid_Loss_list.append(test_loss)
    Valid_Accuracy_list.append(acc)
    list.append(test_loss)


