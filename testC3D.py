import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  

class C3D(nn.Module):  
    def __init__(self):  
        super(C3D, self).__init__()  
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1))  
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))  
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  
        self.fc1 = nn.Linear(128 * 2 * 2 * 2, 128)  
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):  
        x = torch.relu(self.conv1(x))  
        x = self.pool1(x)  
        x = torch.relu(self.conv2(x))  
        x = self.pool2(x)  
        x = x.view(-1, 128 * 2 * 2 * 2)  
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x  

class VideoDataset(Dataset):  
    def __init__(self, video_paths, labels):  
        self.video_paths = video_paths  
        self.labels = labels  

    def __getitem__(self, index):  
        video = torch.load(self.video_paths[index])  
        label = self.labels[index]  
        return video, label  

    def __len__(self):  
        return len(self.video_paths)  

# Tạo dataset và dataloader  
video_paths = [...]  
labels = [...]  
dataset = VideoDataset(video_paths, labels)  
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  

# Khởi tạo mô hình và optimizer  
model = C3D()  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  

# Huấn luyện mô hình  
for epoch in range(10):  
    for batch in dataloader:  
        videos, labels = batch  
        outputs = model(videos)  
        loss = criterion(outputs, labels)  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')