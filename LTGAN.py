import argparse
import os
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.autograd import Variable
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsRegressor
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--kt', type=float, default=0, help='kt parameters')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameters')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=183, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--dpt', type=str, default='', help='load discriminator model')
parser.add_argument('--gpt', type=str, default='', help='load generator model')
parser.add_argument('--train', help='train the network', action='store_true')
parser.add_argument('--impute', help='do imputation', action='store_true')
parser.add_argument('--sim_size', type=int, default=200, help='number of sim_imgs in each type')
parser.add_argument('--file_d', type=str, default='', help='path of data file')
parser.add_argument('--file_c', type=str, default='', help='path of cls file')
parser.add_argument('--ncls', type=int, default=9, help='number of clusters')
parser.add_argument('--knn_k', type=int, default=9, help='neighours used')
parser.add_argument('--lr_rate', type=int, default=9, help='rate for slow learning')
parser.add_argument('--threshold', type=float, default=0.01, help='the convergence threshold')
parser.add_argument('--job_name', type=str, default="", help='the user-defined job name, which will be used to name the output files.')
parser.add_argument('--outdir', type=str, default=".", help='the directory for output.')

opt = parser.parse_args()
max_ncls = opt.ncls

job_name = opt.job_name
GANs_models = opt.outdir + '/GANs_models'
if job_name == "":
    job_name = os.path.basename(opt.file_d) + "-" + os.path.basename(opt.file_c)
model_basename = job_name + "-" + str(opt.latent_dim) + "-" + str(opt.n_epochs) + "-" + str(opt.ncls)
if not os.path.isdir(GANs_models):
    os.makedirs(GANs_models)

img_shape = (opt.channels, opt.img_size, opt.img_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cuda = True if torch.cuda.is_available() else False

class MyDataset(Dataset):
    def __init__(self, d_file, cls_file, transform=None):
        self.data = pd.read_csv(d_file, header=0, index_col=0)
        d = pd.read_csv(cls_file, header=None, index_col=False)
        self.data_cls = pd.Categorical(d.iloc[:, 0]).codes
        self.transform = transform
        self.fig_h = opt.img_size

        # Debugging: Checking data shape and class length
        print(f"Data shape: {self.data.shape}")
        print(f"Data class length: {len(self.data_cls)}")

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        if idx >= self.data.shape[1]:
            raise IndexError("Index out of bounds for data columns.")
        data = self.data.iloc[:, idx].values[0:(self.fig_h * self.fig_h),].reshape(self.fig_h, self.fig_h, 1).astype('double')
        label = np.array(self.data_cls[idx]).astype('int32')
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        data = data.transpose((2, 0, 1))
        return {'data': torch.from_numpy(data), 'label': torch.from_numpy(label)}

def one_hot(batch, depth):
    ones = torch.eye(depth)
    batch = batch.clamp(0, depth - 1)  # Ensure all batch values are within valid range
    return ones.index_select(0, batch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 4
        self.cn1 = 32
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1 * (self.init_size ** 2)))
        self.l1p = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1 * (opt.img_size ** 2)))

        self.conv_blocks_01p = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.ReLU(),
        )

        self.conv_blocks_02p = nn.Sequential(
            nn.Upsample(scale_factor=opt.img_size),
            nn.Conv2d(max_ncls, self.cn1 // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(40, 0.8),
            nn.Conv2d(40, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, opt.channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, noise, label_oh):
        out = self.l1p(noise)
        out = out.view(out.shape[0], self.cn1, opt.img_size, opt.img_size)
        out01 = self.conv_blocks_01p(out)

        label_oh = label_oh.unsqueeze(2).unsqueeze(2)
        out02 = self.conv_blocks_02p(label_oh)

        out1 = torch.cat((out01, out02), 1)
        out1 = self.conv_blocks_1(out1)
        return out1

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cn1 = 32
        self.down_size0 = 64
        self.down_size = 32
        self.pre = nn.Sequential(nn.Linear(opt.img_size ** 2, self.down_size0 ** 2))
        self.down = nn.Sequential(
            nn.Conv2d(opt.channels, self.cn1, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, self.cn1 // 2, 3, 1, 1),
            nn.BatchNorm2d(self.cn1 // 2),
            nn.ReLU(),
        )

        self.conv_blocks02p = nn.Sequential(
            nn.Upsample(scale_factor=self.down_size),
            nn.Conv2d(max_ncls, self.cn1 // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        down_dim = 24 * (self.down_size) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.ReLU(),
            nn.Linear(16, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU()
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, opt.channels, 2, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img, label_oh):
        out00 = self.pre(img.view((img.size()[0], -1))).view((img.size()[0], 1, self.down_size0, self.down_size0))
        out01 = self.down(out00)

        label_oh = label_oh.unsqueeze(2).unsqueeze(2)
        out02 = self.conv_blocks02p(label_oh)

        out1 = torch.cat((out01, out02), 1)
        out = self.fc(out1.view(out1.size(0), -1))
        out = self.up(out.view(out.size(0), 24, self.down_size, self.down_size))
        return out

def my_dbscan_impute(data_imp_org_k, sim_out_k, eps=0.5, min_samples=10):
    sim_size = sim_out_k.shape[0]
    out = data_imp_org_k.copy()
    q1k = data_imp_org_k.reshape((opt.img_size * opt.img_size, 1))
    q1kl = np.int8(q1k > 0)
    q1kn = np.repeat(q1k * q1kl, repeats=sim_size, axis=1)
    sim_out_tmp = sim_out_k.reshape((sim_size, opt.img_size * opt.img_size)).T
    sim_outn = sim_out_tmp * np.repeat(q1kl, repeats=sim_size, axis=1)
    diff = q1kn - sim_outn
    diff = diff * diff
    rel = np.sum(diff, axis=0)
    locs = np.where(q1kl == 0)[0]
    sim_out_c = np.median(sim_out_tmp[:, rel.argsort()[0:min_samples]], axis=1)
    out[locs] = sim_out_c[locs]
    return out

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

transformed_dataset = MyDataset(d_file=opt.file_d, cls_file=opt.file_c, transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

gamma = opt.gamma
lambda_k = 0.001
k = opt.kt

if opt.train:
    model_exists = os.path.isfile(GANs_models + '/' + model_basename + '-g.pt')
    if model_exists:
        overwrite = input("WARNING: A trained model exists with the same settings for your data.\nDo you want to train and overwrite it?: (y/n)\n")
        if overwrite != "y":
            print("The training was deprecated since optical model exists.")
            print("scIGANs continues imputation using existing model...")
            sys.exit()
    print("The optimal model will be output in \"" + os.getcwd() + "/" + GANs_models + "\" with basename = " + model_basename)

    max_M = sys.float_info.max
    min_dM = 0.001
    dM = 1
    for epoch in range(opt.n_epochs):
        cur_M = 0
        cur_dM = 1
        for i, batch_sample in enumerate(dataloader):
            imgs = batch_sample['data'].type(Tensor).to(device)
            label = batch_sample['label'].to(device)
            label_oh = one_hot((label).type(torch.LongTensor), max_ncls).type(Tensor).to(device)

            real_imgs = Variable(imgs.type(Tensor))

            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(device)
            gen_imgs = generator(z, label_oh)
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs, label_oh) - gen_imgs))
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            d_real = discriminator(real_imgs, label_oh)
            d_fake = discriminator(gen_imgs.detach(), label_oh)
            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - k * d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            diff = torch.mean(gamma * d_loss_real - d_loss_fake)
            k = k + lambda_k * np.ndarray.item(diff.detach().data.cpu().numpy())
            k = min(max(k, 0), 1)
            M = (d_loss_real + torch.abs(diff)).item()
            cur_M += M
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, delta_M: %f,k: %f" % (
            epoch + 1, opt.n_epochs, i + 1, len(dataloader), np.ndarray.item(d_loss.detach().data.cpu().numpy()), np.ndarray.item(
                g_loss.detach().data.cpu().numpy()), M, dM, k))
            sys.stdout.flush()
            batches_done = epoch * len(dataloader) + i
        cur_M = cur_M / len(dataloader)
        if cur_M < max_M:
            torch.save(discriminator.state_dict(), GANs_models + '/' + model_basename + '-d.pt')
            torch.save(generator.state_dict(), GANs_models + '/' + model_basename + '-g.pt')
            dM = min(max_M - cur_M, cur_M)
            if dM < min_dM:
                print("Training was stopped after " + str(epoch + 1) + " epochs since the convergence threshold (" + str(
                    min_dM) + ".) reached: " + str(dM))
                break
            cur_dM = max_M - cur_M
            max_M = cur_M
        if epoch + 1 == opt.n_epochs and cur_dM > min_dM:
            print("Training was stopped after " + str(epoch + 1) + " epochs since the maximum epochs reached: " + str(
                opt.n_epochs) + ".")
            print("WARNING: the convergence threshold (" + str(min_dM) + ") was not met. Current value is: " + str(cur_dM))
            print("You may need more epochs to get the most optimal model!!!")

if opt.impute:
    if opt.gpt == '':
        model_g = GANs_models + '/' + model_basename + '-g.pt'
        model_exists = os.path.isfile(model_g)
        if not model_exists:
            print("ERROR: There is no model exists with the given settings for your data.")
            print("Please set --train instead of --impute to train a model first.")
            sys.exit("scIGANs stopped!!!")
    else:
        model_g = opt.gpt
    print(model_g + " is used for imputation.")
    if cuda:
        generator.load_state_dict(torch.load(model_g))
    else:
        generator.load_state_dict(torch.load(model_g, map_location=lambda storage, loc: storage))
    sim_size = opt.sim_size
    sim_out = list()
    for i in range(opt.ncls):
        label_oh = one_hot(torch.from_numpy(np.repeat(i, sim_size)).type(torch.LongTensor), max_ncls).type(Tensor).to(
            device)
        z = Variable(Tensor(np.random.normal(0, 1, (sim_size, opt.latent_dim)))).to(device)
        fake_imgs = generator(z, label_oh).detach().cpu().numpy()
        sim_out.append(fake_imgs)
    mydataset = MyDataset(d_file=opt.file_d, cls_file=opt.file_c)
    data_imp_org = np.asarray(
        [mydataset[i]['data'].reshape((opt.img_size * opt.img_size)) for i in range(len(mydataset))]).T

    # 调试：打印数据长度
    print(f"Data_imp_org shape: {data_imp_org.shape}")
    print(f"Mydataset length: {len(mydataset)}")

    data_imp = data_imp_org.copy()
    sim_out_org = sim_out
    rels = []

    # 使用孤立森林检测并移除异常值
    isolation_forest = IsolationForest(contamination=0.25)
    for k in range(len(mydataset)):
        label = int(mydataset[k]['label'])
        if label >= len(sim_out_org):
            raise IndexError(f"Label {label} is out of bounds for sim_out_org with length {len(sim_out_org)}")
        sample_data = data_imp_org[:, k].reshape(1, -1)
        outlier_pred = isolation_forest.fit_predict(sample_data)
        if outlier_pred[0] == -1:
            data_imp[:, k] = np.nan

    for k in range(len(mydataset)):
        label = int(mydataset[k]['label'])
        if label >= len(sim_out_org):
            raise IndexError(f"Label {label} is out of bounds for sim_out_org with length {len(sim_out_org)}")
        rel = my_dbscan_impute(data_imp_org[:, k], sim_out_org[label], eps=0.5, min_samples=opt.knn_k)
        rels.append(rel)
    from sklearn.preprocessing import StandardScaler

    imputed_data = np.array(rels)
    scaler = StandardScaler()
    imputed_data_scaled = scaler.fit_transform(imputed_data)

    pd.DataFrame(rels).to_csv(os.path.dirname(os.path.abspath(opt.file_d)) + '/scIGANs-' + job_name + '.csv')

    # 禁用 cuDNN 优化
    torch.backends.cudnn.enabled = False

    # LSTM 模块定义
    class LSTMModule(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, batch_first=True):
            super(LSTMModule, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, input_data):
            lstm_out, _ = self.lstm(input_data.unsqueeze(1))
            lstm_out = lstm_out.squeeze(1)
            lstm_out = self.fc(lstm_out)
            return lstm_out


    # 修改 LSTM hidden_size 和 output_size
    lstm_model = LSTMModule(input_size=opt.img_size * opt.img_size, hidden_size=64,
                            output_size=opt.img_size * opt.img_size).to(device)

    # 定义优化器和损失函数
    criterion = nn.SmoothL1Loss()  # Huber Loss
  # 使用均方误差损失函数
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.00001)  # 使用Adam优化器

    # 准备 LSTM 训练数据
    imputed_data_tensor = torch.tensor(imputed_data_scaled, dtype=torch.float32).to(device)  # 将数据转换为tensor并放入GPU

    # 梯度累积设置
    accumulation_steps = 8  # 设置累积的步数
    batch_size_lstm = 512  # 分批次进行LSTM训练，减少每次训练时的内存占用

    # 训练 LSTM 模型
    lstm_epochs = 1000  # 设置 LSTM 训练的 epoch 数
    num_batches = int(np.ceil(imputed_data_tensor.size(0) / batch_size_lstm))

    for epoch in range(lstm_epochs):
        lstm_model.train()
        optimizer_lstm.zero_grad()

        for i in range(num_batches):
            start_idx = i * batch_size_lstm
            end_idx = min((i + 1) * batch_size_lstm, imputed_data_tensor.size(0))

            # 批量数据
            batch_data = imputed_data_tensor[start_idx:end_idx]

            # 前向传播
            output = lstm_model(batch_data)
            loss = criterion(output, batch_data)  # 使用自监督损失

            # 反向传播和梯度累积
            loss = loss / accumulation_steps
            loss.backward()

            # 每 accumulation_steps 次更新一次权重
            if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
                optimizer_lstm.step()
                optimizer_lstm.zero_grad()

        print(f"Epoch [{epoch + 1}/{lstm_epochs}], Loss: {loss.item()}")

    # 使用训练好的 LSTM 进行调优
    lstm_model.eval()
    lstm_adjusted_data = []
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size_lstm
            end_idx = min((i + 1) * batch_size_lstm, imputed_data_tensor.size(0))

            # 批量数据
            batch_data = imputed_data_tensor[start_idx:end_idx]

            # 前向传播
            lstm_adjusted_batch = lstm_model(batch_data).cpu().detach().numpy()
            lstm_adjusted_data.append(lstm_adjusted_batch)

    # 将分批次的结果合并
    lstm_adjusted_data = np.concatenate(lstm_adjusted_data, axis=0)
    
    print("Replacing negative values using KNN interpolation...")

    # Flatten the data for processing
    flat_data = lstm_adjusted_data.reshape(-1, opt.img_size * opt.img_size)

    # Create a mask of negative values
    neg_mask = flat_data < 0

    # For each feature (column), apply KNN interpolation for negative values
    for feature_idx in range(flat_data.shape[1]):
        feature_column = flat_data[:, feature_idx]
        negative_idx = np.where(neg_mask[:, feature_idx])[0]

        # If there are negative values, apply KNN interpolation
        if len(negative_idx) > 0:
            non_negative_idx = np.where(~neg_mask[:, feature_idx])[0]

            # Check how many non-negative samples are available
            n_non_negative = len(non_negative_idx)

            if n_non_negative > 0:
                # Set n_neighbors to the minimum between knn_k and available non-negative samples
                n_neighbors = min(opt.knn_k, n_non_negative)

                # Prepare the KNN regressor
                knn = KNeighborsRegressor(n_neighbors=n_neighbors)

                # Fit KNN on non-negative values
                knn.fit(non_negative_idx.reshape(-1, 1), feature_column[non_negative_idx])

                # Predict the negative values using KNN
                predicted_values = knn.predict(negative_idx.reshape(-1, 1))

                # Replace the negative values with the predicted ones
                flat_data[negative_idx, feature_idx] = predicted_values
            else:
                # Handle cases where there are no non-negative samples to use for KNN
                print(f"Warning: No non-negative samples available for feature {feature_idx}. Skipping KNN interpolation.")
                continue

    # Reshape the data back to original dimensions
    lstm_adjusted_data = flat_data.reshape(-1, opt.img_size, opt.img_size)


    # Save the KNN-imputed data to CSV
    lstm_adjusted_df = pd.DataFrame(lstm_adjusted_data.reshape(len(lstm_adjusted_data), -1))
    lstm_adjusted_csv_path = os.path.join(opt.outdir, 'scIGANs-LSTM-KNN-' + job_name + '.csv')
    lstm_adjusted_df.to_csv(lstm_adjusted_csv_path)

    print(f"LSTM 调优后的 CSV 已保存至: {lstm_adjusted_csv_path}")
