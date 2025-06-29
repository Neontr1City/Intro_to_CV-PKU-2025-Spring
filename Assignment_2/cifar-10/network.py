import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        pass
        # ----------TODO------------
        # define a network 
        #conv1->conv2->pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #conv3->conv4->pool
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        #self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        #fc1->fc2
        self.fc1 = nn.Linear(64*8*8, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_class)
        
        # ----------TODO------------

    def forward(self, x):

        # ----------TODO------------
        # network forwarding 
        
        #block1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        #block2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        #global_pool flatten
        #x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        #fc
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        # ----------TODO------------

        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 
