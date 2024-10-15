import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from wave import wavelet_transform

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim//2,input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        x1, x2 = wavelet_transform(x)# 离散小波变化提取高频部分
        x2 = torch.from_numpy(x2).float()
        encoded = self.encoder(x2)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def train(self, input_data, num_epochs=100, learning_rate=0.001):
        input_data_tensor = torch.from_numpy(input_data).float()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            _, decoded = self.forward(input_data_tensor)
            loss = criterion(decoded, input_data_tensor)
            loss.backward()
            optimizer.step()

    # def generate_abnormal_features(self, num_samples):
    #     latent_dim = self.encoder[4].out_features
    #     noise = torch.randn(num_samples, latent_dim)
    #     abnormal_features = self.decoder(noise)
    #     return abnormal_features.detach().numpy()

    def generate_abnormal_features(self, input_data, num_samples, noise_std=0.1):
        input_data_tensor = torch.from_numpy(input_data).float()
        # print("input_data_tensor.shape:",input_data_tensor.shape)
        encoded_data, _ = self.forward(input_data_tensor)

        # 从编码数据中随机选择指定数量的样本
        indices = np.random.choice(encoded_data.size(0), num_samples, replace=True)
        selected_encoded_data = encoded_data[indices]

        # 生成随机噪声
        noise = torch.randn(selected_encoded_data.size()) * noise_std

        # 使用解码器生成新样本，并添加噪声
        generated_samples = self.decoder(selected_encoded_data + noise).detach().numpy()
        return generated_samples

# 示例用法
input_dim = 128
encoding_dim = 32
num_samples = 100

autoencoder = Autoencoder(input_dim, encoding_dim)
normal_features = np.random.randn(num_samples, input_dim)
# print("normal_features.shape:", normal_features.shape)

autoencoder.train(normal_features, num_epochs=100, learning_rate=0.001)

generated_abnormal_features = autoencoder.generate_abnormal_features(normal_features,num_samples)
# print('generated_abnormal_features.shape:', generated_abnormal_features.shape)