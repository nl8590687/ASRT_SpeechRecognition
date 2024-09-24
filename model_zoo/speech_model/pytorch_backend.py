import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
class SpeechModel251BN(nn.Module):
    def __init__(self,input_shape: tuple = (1600, 200, 1), output_size: int = 1428):
        super(SpeechModel251BN, self).__init__()

        self.input_shape = input_shape
        self._pool_size = 8
        self._model_name = 'SpeechModel251bn'
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        
        self.conv0 = nn.Conv2d(1, 32, kernel_size=(3, 3),padding='same')
        self.bn0 = nn.BatchNorm2d(32,eps=0.0002)

        self.conv1 = nn.Conv2d(32, 32, kernel_size=(3, 3),padding='same')
        self.bn1 = nn.BatchNorm2d(32,eps=0.0002)
        
        # self.maxpool1 = F.max_pool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3),padding='same')
        self.bn2 = nn.BatchNorm2d(64,eps=0.0002)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3),padding='same')
        self.bn3 = nn.BatchNorm2d(64,eps=0.0002)
        
        # self.maxpool2 = F.max_pool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3),padding='same')
        self.bn4 = nn.BatchNorm2d(128,eps=0.0002)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3),padding='same')
        self.bn5 = nn.BatchNorm2d(128,eps=0.0002)
        
        # self.maxpool3 = F.max_pool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3),padding='same')
        self.bn6 = nn.BatchNorm2d(128,eps=0.0002)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=(3, 3),padding='same')
        self.bn7 = nn.BatchNorm2d(128,eps=0.0002)

        # self.maxpool4 = F.max_pool2d(kernel_size=(1,1),stride=(1,1))

        self.conv8 = nn.Conv2d(128, 128, kernel_size=(3, 3),padding='same')
        self.bn8 = nn.BatchNorm2d(128,eps=0.0002)

        self.conv9 = nn.Conv2d(128, 128, kernel_size=(3, 3),padding='same')
        self.bn9 = nn.BatchNorm2d(128,eps=0.0002)

        # self.maxpool5 = F.max_pool2d(kernel_size=(1,1),stride=(1,1))

        self.dense0 = nn.Linear(input_shape[1]//8*128, 128)

        self.dense1 = nn.Linear(128, output_size)

        self.ctc_loss = nn.CTCLoss(blank=0)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size())
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.size())
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        # print(x.size())
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        # print(x.size())
        x = F.max_pool2d(x, kernel_size=(1, 1), stride=(1, 1))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        # print(x.size())
        x = F.max_pool2d(x, kernel_size=(1, 1), stride=(1, 1))
        # print(x.size())
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0),x.size(1), -1)
        # x = x.permute(0, 2, 1)
        # x = x.view(x.size(0),x.size(1), -1)
        # print(x.size())
        x = F.relu(self.dense0(x))
        # print(x.size())
        x = F.softmax(self.dense1(x))
        # print(x.size())
        return x
    def compute_loss(self, y_pred,labels,input_length,label_length):
        y_pred = y_pred.permute(1, 0, 2)
        loss = self.ctc_loss(y_pred, labels, input_length, label_length)
        return loss
    def train_model(self, train_loader, optimizer, num_epochs=10, device='cpu'):
        self.to(device)
        self.train()

        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0.0
            for batch in train_loader:
                inputs, labels, input_lengths, label_lengths = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                input_lengths = input_lengths.to(device)
                label_lengths = label_lengths.to(device)

                optimizer.zero_grad()
                y_pred = self.forward(inputs)

                loss = self.compute_loss(y_pred, labels, input_lengths, label_lengths)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # for epoch in range(num_epochs):
        #     epoch_loss = 0.0

        #     for batch in train_loader:
        #         inputs, labels, input_lengths, label_lengths = batch
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         input_lengths = input_lengths.to(device)
        #         label_lengths = label_lengths.to(device)

        #         optimizer.zero_grad()
        #         y_pred = self.forward(inputs)

        #         loss = self.compute_loss(y_pred, labels, input_lengths, label_lengths)
        #         loss.backward()
        #         optimizer.step()

        #         epoch_loss += loss.item()

        #     avg_loss = epoch_loss / len(train_loader)
        #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    def get_model(self):
        return self
    def get_model_name(self) -> str:
        return self._model_name