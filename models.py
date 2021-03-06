from torch import nn
import torch
import torch.nn.functional as F


class DCDiscriminator(nn.Module):

    def __init__(self, img_dim, num_conv1, size_conv1, num_conv2, size_conv2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, num_conv1, size_conv1, 1)
        self.conv2 = nn.Conv2d(num_conv1, num_conv2, size_conv2, 1)

        self.conv2_out_w = (1 + ((1 + (img_dim - size_conv1)) // 2 - size_conv2)) // 2
        self.fc1_in = self.conv2_out_w * self.conv2_out_w * num_conv2

        self.fc1 = nn.Linear(self.fc1_in, self.fc1_in)
        self.fc2 = nn.Linear(self.fc1_in, 1)


    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = torch.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = torch.max_pool2d(x, 2)

        x = x.view(-1, self.fc1_in)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)

        return x


class DCGenerator(nn.Module):

    def __init__(self, noise_dim, hidden_dim, num_conv1, size_conv1, num_conv2, size_conv2, img_dim):
        super().__init__()
        self.conv2_in_w = 1 + (img_dim - size_conv2 + 2) // 2
        self.conv1_in_w = 1 + (self.conv2_in_w - size_conv1 + 2) // 2
        self.fc2_in = self.conv1_in_w * self.conv1_in_w * num_conv1
        self.num_conv1 = num_conv1

        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, self.fc2_in)
        self.bn2 = nn.BatchNorm1d(self.fc2_in)

        self.deconv1 = nn.ConvTranspose2d(num_conv1, num_conv2, size_conv1, 2, 1)
        self.bn3 = nn.BatchNorm2d(num_conv2)
        self.deconv2 = nn.ConvTranspose2d(num_conv2, 1, size_conv2, 2, 1)


    def forward(self, z):

        x = self.fc1(z)
        x = torch.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn2(x)

        x = x.view(-1, self.num_conv1, self.conv1_in_w, self.conv1_in_w)
        x = self.deconv1(x)
        x = torch.relu(x)

        x = self.bn3(x)
        x = self.deconv2(x)
        x = torch.tanh(x)

        return x


class cDCDiscriminator(DCDiscriminator):
    def __init__(self, img_dim, num_conv1, size_conv1, num_conv2, size_conv2, label_emb_dim, n_classes):
        super().__init__(img_dim, num_conv1, size_conv1, num_conv2, size_conv2)
        self.label_emb_dim = label_emb_dim
        self.n_classes = n_classes
        self.emb = nn.Embedding(self.n_classes, self.label_emb_dim)
        self.label_fc = nn.Linear(self.label_emb_dim, img_dim**2)
        self.conv1 = nn.Conv2d(1+1, num_conv1, size_conv1, 1)

    def forward(self, x, label):
        label_emb = self.emb(label)
        label_fc = self.label_fc(label_emb)
        label_fc = label_fc.view(-1, 1, *x.shape[2:])
        x = torch.cat((x, label_fc), dim=1)

        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        # x = self.bn1(x)
        x = torch.max_pool2d(x, 2)

        # x = torch.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        # x = self.bn2(x)
        x = torch.max_pool2d(x, 2)

        # x = torch.max_pool2d(x, 2)

        x = x.view(-1, self.fc1_in)
        x = self.fc2(x)

        return x



class cDCGenerator(DCGenerator):
    def __init__(self, noise_dim, hidden_dim, num_conv1, size_conv1, num_conv2, size_conv2, img_dim, label_emb_dim,
                 n_classes):
        super().__init__(noise_dim, hidden_dim, num_conv1, size_conv1, num_conv2, size_conv2, img_dim)
        self.label_emb_dim = label_emb_dim
        self.n_classes = n_classes
        self.emb = nn.Embedding(self.n_classes, self.label_emb_dim)
        self.label_fc = nn.Linear(self.label_emb_dim, self.conv1_in_w * self.conv1_in_w)
        self.deconv1 = nn.ConvTranspose2d(num_conv1+1, num_conv2, size_conv1, 2, 1)

    def forward(self, z, label):
        label_emb = self.emb(label)
        label_fc = self.label_fc(label_emb)
        label_fc = label_fc.view(-1, 1, self.conv1_in_w, self.conv1_in_w)

        x = self.fc1(z)
        x = torch.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn2(x)

        x = x.view(-1, self.num_conv1, self.conv1_in_w, self.conv1_in_w)
        x = torch.cat((x, label_fc), dim=1)
        x = self.deconv1(x)
        x = torch.relu(x)
        x = self.bn3(x)

        x = self.deconv2(x)
        x = torch.tanh(x)

        return x

