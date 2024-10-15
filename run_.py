from sklearn.metrics import roc_auc_score,average_precision_score
import os
import shutil
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics
from typing import Any, Optional, Tuple
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import time


def divide_infor_label(data):
    link_label = data[:, 0]

    infor = data[:, 1:]
    return link_label, infor


def divide_network_edge(data):
    generate_label = data[:, :, 0]
    timestamp_label = data[:, :, 1]
    edge = data[:, :, 2:]
    return timestamp_label, generate_label, edge


def get_train_test(train_data, test_data, valid_data, batch_size):
    # Preprocessing
    train = pd.read_csv(train_data, header=None, sep=',')
    test = pd.read_csv(test_data, header=None, sep=',')
    valid = pd.read_csv(valid_data, header=None, sep=',')

    train = np.array(train)
    test = np.array(test)
    valid = np.array(valid)

    train_link_label, train = divide_infor_label(train)
    test_link_label, test = divide_infor_label(test)
    valid_link_label, valid = divide_infor_label(valid)

#    rus = RandomUnderSampler(random_state=0, replacement=True)
#    enn = EditedNearestNeighbours()
#
#    train, train_link_label = rus.fit_resample(train, train_link_label)

    train = torch.from_numpy(train).unsqueeze(dim=1).float()
    train_link_label = np.array(train_link_label)
    train_link_label = torch.from_numpy(train_link_label).unsqueeze(dim=1).long()
    train_set = TensorDataset(train, train_link_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test = torch.from_numpy(test).unsqueeze(dim=1).float()
    test_link_label = np.array(test_link_label)
    test_link_label = torch.from_numpy(test_link_label).unsqueeze(dim=1).long()
    test_set = TensorDataset(test, test_link_label)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    valid = torch.from_numpy(valid).unsqueeze(dim=1).float()
    valid_link_label = np.array(valid_link_label)
    valid_link_label = torch.from_numpy(valid_link_label).unsqueeze(dim=1).long()
    valid_set = TensorDataset(valid, valid_link_label)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # unique, counts = np.unique(train_link_label, return_counts=True)
    # print("Train labels distribution:", dict(zip(unique, counts)))
#    from collections import Counter
    # unique, counts = np.unique(test_link_label, return_counts=True)
    # print("Test labels distribution:", dict(zip(unique, counts)))
#    print("valid under sampling results: ", sorted(Counter(valid_link_label).items()))
    # unique, counts = np.unique(valid_link_label, return_counts=True)
    # print("Valid labels distribution:", dict(zip(unique, counts)))

    return train_loader, test_loader, valid_loader


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)


import torch
import torch.nn as nn
from torch.autograd import Function


class Adversarial(nn.Module):
    def __init__(self, in_dim, network_numbers):
        super(Adversarial, self).__init__()

        # Add extra convolutional layers
        self.generality_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()

        )

        self.target_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.sample_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1))

        self.weight_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim + 2, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.weight_softmax = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax(dim=1))

        # Add extra linear layers
        self.link_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1))

        self.network_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, network_numbers),
            nn.Softmax(dim=1))

        self.residual1 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.residual2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, edge_embbing, weight_input, coeff=10):
#        print("edge_embbing.shape:",edge_embbing.shape)
        edge_embbing = edge_embbing.permute(0, 2, 1)

        generality_feature = self.generality_conv(edge_embbing)
        generality_feature = self.residual1(generality_feature) + generality_feature  # Add residual connection

        generality_feature = generality_feature.view(generality_feature.size(0), -1)

        target_feature = self.target_conv(edge_embbing)
        target_feature = self.residual2(target_feature) + target_feature  # Add residual connection

        target_feature = target_feature.view(target_feature.size(0), -1)

        weight_input = weight_input.permute(0, 2, 1)
        weight_out = self.weight_conv(weight_input)
        weight_out = weight_out.view(weight_out.size(0), -1)
        weight_out = self.weight_softmax(weight_out)

        # feature = torch.zeros_like(target_feature)
        # for i in range(feature.shape[0]):
        #     feature[i] = generality_feature[i] * weight_out[i][0] + target_feature[i] * weight_out[i][1]
        feature=0.3*generality_feature+0.7*target_feature

        link_output = self.link_classifier(feature)
        # reverse_feature = grad_reverse(feature, coeff)
        network_output = self.network_classifier(feature)
        sample_output = self.sample_classifier(feature)

        return link_output, network_output, sample_output


def get_pred(out):
    out = out.argmax(dim=1)  # 取出每行的最大值索引
    one = torch.ones_like(out)
    zero = torch.zeros_like(out)
    out = torch.where(out == 1, one, zero)  # 将最大值索引为1的位置置为1,其余置为0
    return out


def get_acc(out, label):
    out = get_pred(out)
    accuracy = (out == label).float().mean()
    return accuracy


def compute_gradient_penalty(model, edge, infor):
    edge.requires_grad_(True)
    infor.requires_grad_(True)

    link_out, network_out = model(edge, infor)

    gradients = torch.autograd.grad(outputs=link_out,
                                    inputs=edge,
                                    grad_outputs=torch.ones(
                                        link_out.size()).cuda() if torch.cuda.is_available() else torch.ones(
                                        link_out.size()),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_Adversarial_Model(dataset, train_loader, valid_loader, model, criterion,initial_learning_rate):
    model_path = 'output1/' + dataset + '_model/'
    if os.path.exists(model_path):  # 清除之前运行代码生成的模型
        shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    best_valid_dir = ''
    best_valid_auc = 0

    total_start_time = time.time()
    for epoch in range(epochs + 1):
        p = epoch / epochs
        # learning_rate = initial_learning_rate / pow((1 + 10 * p), 0.75)
        learning_rate = initial_learning_rate

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        task_model = Adversarial(in_dim=128, network_numbers=network_number)
        task_model.load_state_dict(model.state_dict())
        inner_optimizer = torch.optim.Adam(task_model.parameters(), lr=learning_rate)
        model.train()
        loss_vec = []
        auc_vec = []
        for support_data in train_loader:
            optimizer.zero_grad()
            infor, link_label = support_data
            # print("infor.shape",infor.shape,type(infor))
            # sdsa
            network_label, sample_label, edge = divide_network_edge(infor)
            if  torch.cuda.is_available():
                infor = infor.cuda()
                edge = edge.cuda()
                link_label = link_label.cuda()
                network_label = network_label.cuda()
                sample_label = sample_label.cuda()
            infor = Variable(infor)
            edge = Variable(edge)
#            print("edge.shape:",edge.shape)
            link_label = Variable(link_label)
            network_label = Variable(network_label)
            sample_label = Variable(sample_label)
            link_out, network_out, sample_out = task_model(edge, infor)
#            print("link_out.shape:",link_out.shape)
#            print("link_label",link_label.shape)
            link_loss = criterion(link_out, link_label.squeeze(1).long())
            # network_loss = criterion(network_out, network_label.squeeze(1).long())
            sample_loss = criterion(sample_out, sample_label.squeeze(1).long())
            loss = link_loss
            loss_vec.append(loss.detach().cpu().numpy())

            try:
                auc = roc_auc_score(link_label.cpu().numpy(), link_out.detach().cpu().numpy()[:, 1])
                auc_vec.append(auc)
            except ValueError:
                pass

            auc = roc_auc_score(link_label.cpu().numpy(), link_out.detach().cpu().numpy()[:, 1])
            auc_vec.append(auc)
            loss.backward(retain_graph=True)
            inner_optimizer.step()
            model.load_state_dict(task_model.state_dict())
        loss = np.mean(loss_vec)
        auc = np.mean(auc_vec)
        valid_auc_vec = []
        for query_data in valid_loader:
            query_infor, query_link_label = query_data
            _, query_sample_label, query_edge = divide_network_edge(query_infor)
            if torch.cuda.is_available():
                with torch.no_grad():
                    query_infor = Variable(query_infor).cuda()
                    query_edge = Variable(query_edge).cuda()
                    query_link_label = Variable(query_link_label).cuda()
                    query_sample_label = Variable(query_sample_label).cuda()

            else:
                with torch.no_grad():
                    query_infor = Variable(query_infor)
                    query_edge = Variable(query_edge)
                    query_link_label = Variable(query_link_label)
                    query_sample_label = Variable(query_sample_label)
            query_link_out,_, query_sample_out = model(query_edge, query_infor)
            # dot = make_dot((link_output, network_output, sample_output), params=dict(model.named_parameters()))
            # dot.format = 'pdf'
            # dot.render("model_structure")
            query_link_loss = criterion(query_link_out, query_link_label.squeeze(1).long())
            query_sample_loss = criterion(query_sample_out, query_sample_label.squeeze(1).long())
            meta_loss = query_link_loss
            meta_loss.backward()
            optimizer.step()
            link_out_np = link_out.detach().cpu().numpy()
            link_label_np = link_label.cpu().numpy()
            # added_fictitious_link = False

            # if len(np.unique(link_label_np)) == 1 and not added_fictitious_link:
            #    link_label_np = np.append(link_label_np, 1)
            #    link_out_np = np.append(link_out_np, [[0.5, 0.7]], axis=0)
            #    link_label_np = np.append(link_label_np, 0)
            #    link_out_np = np.append(link_out_np, [[0.7, 0.5]], axis=0)
            #    added_fictitious_link = True
            try:
                valid_auc = metrics.roc_auc_score(link_label_np, link_out_np[:, 1])
                valid_auc_vec.append(valid_auc)
            except ValueError:
                pass
            # valid_auc = roc_auc_score()
            # valid_auc_vec.append(valid_auc)

        valid_auc = np.mean(valid_auc_vec)


        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_valid_dir = model_path + 'model' + str(epoch) + '.pkl'
            torch.save(model.state_dict(), best_valid_dir)

#        if epoch % 10 == 0:
        print(
                'Adversarial Model Epoch: [{}/{}], learning rate:{:.6f}, train loss:{:.4f}, train auc:{:.4f}, valid auc:{:.4f}'.format(
                    epoch, epochs, learning_rate, loss, auc, best_valid_auc))
    total_end_time = time.time()

    # 计算总的训练时间
    total_elapsed_time = total_end_time - total_start_time

    print('Total training time: {:.2f} seconds'.format(total_elapsed_time))
    return best_valid_dir

from torchviz import make_dot
def test_Adversarial_Model(test_loader, adversarial_model, best_valid_dir):
    adversarial_model.load_state_dict(torch.load(best_valid_dir))
    adversarial_model.eval()

    acc_vec = []
    precision_vec = []
    f1_vec = []
    auc_vec = []
    auc_vec1 = []
    y_score = []
    y_true = []
    aupr_vec=[]
    for i, data in enumerate(test_loader):
        infor, link_label = data
        _, _, edge = divide_network_edge(infor)
        if torch.cuda.is_available():
            with torch.no_grad():
                infor = Variable(infor).cuda()
                edge = Variable(edge).cuda()
                link_label = Variable(link_label).cuda()
        else:
            with torch.no_grad():
                infor = Variable(infor)
                edge = Variable(edge)
                link_label = Variable(link_label)

        adversarial_out, _, _ = adversarial_model(edge, infor)

        # dot = make_dot((adversarial_out), params=dict(adversarial_model.named_parameters()))
        # dot.format = 'pdf'
        # dot.render("model_structure")
        pred = get_pred(adversarial_out).cpu()
        link_label = link_label.squeeze(1).long().cpu()

        acc = (pred == link_label).float().mean()
        acc_vec.append(acc.detach().cpu().numpy())
        score = adversarial_out[:, 1].cpu().detach()
        if i == 0:
            y_score = score.data.cpu().numpy()
            y_true = link_label.data.cpu().numpy()
        else:
            y_score = np.concatenate((y_score, score.data.cpu().numpy()), axis=0)
            y_true = np.concatenate((y_true, link_label.data.cpu().numpy()), axis=0)
        precision = metrics.precision_score(link_label, pred, average='weighted')
        f1 = metrics.f1_score(link_label,pred, average='weighted')
        precision_vec.append(precision)
        f1_vec.append(f1)
        aupr1=average_precision_score(link_label, pred)
        aupr_vec.append(aupr1)
        try:
            auc = metrics.roc_auc_score(link_label, pred)
            auc_vec.append(auc)
        except ValueError:
            pass
        try:
            auc1 = roc_auc_score(link_label, adversarial_out.detach().cpu().numpy()[:, 1])
            auc_vec1.append(auc1)
        except ValueError:
            pass
    auc = np.mean(auc_vec)

    precision = np.mean(precision_vec)
    accuracy = np.mean(acc_vec)
    f1_score = np.mean(f1_vec)
    auc1 = np.mean(auc_vec1)
    aupr11=np.mean(aupr_vec)
    return auc1, precision, accuracy, f1_score, aupr11

#
#def run_Adversarial_model(dataset, train_loader, test_loader, valid_loader, network_numbers):
#    adversarial_model = Adversarial(in_dim=128, network_numbers=network_numbers)
#    # if torch.cuda.is_available():
#    #    adversarial_model = adversarial_model.cuda()
#    criterion = nn.CrossEntropyLoss()
#    best_valid_dir = train_Adversarial_Model(dataset, train_loader, valid_loader, adversarial_model, criterion)
#
#    # auc, precision, acc, f1, aupr = test_Adversarial_Model(valid_loader, adversarial_model, best_valid_dir)
#    auc, precision, acc, f1, aupr = test_Adversarial_Model(valid_loader, adversarial_model, '/output/')
#    return auc, precision, acc, f1, aupr
batch_size = 64
initial_learning_rate =0.00001
epochs = 5
repeats= 10
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
CUDA_LAUNCH_BLOCKING=1
outputpath = './output1/'
if not os.path.exists(outputpath):
    os.mkdir(outputpath)
outfile = open('./output1/out.txt', 'w', encoding='utf-8')
dataset = 'reddit'
#fold=1
network_number=5
#train_data = f'./GNU b/{dataset}_train_fold_{fold}_with_labels.txt'
#test_data = f'./GNU b/{dataset}_test_fold_{fold}_with_labels.txt'
#train_data = f'./disneg_data/train2_004_disneg.txt'
#test_data = f'./disneg_data/test2_004_disneg.txt'
#train_data = f'./newdata/train3_004.txt'
#test_data = f'./newdata/test3_004.txt'
# train_data = f'train3_sup_disneg.txt'
# valid_data = f'valid3_sup_disneg.txt'
# test_data=f'test3_sup_disneg.txt'
dataname=dataset
train_data = f''+dataname+'train.txt'
valid_data = f''+dataname+'valid.txt'
test_data=f''+dataname+'test.txt'

# print('Target layer filename:', train_data, '---')
# print('Auxiliary layer filename:', test_data, '---')
acc_t = []
precision_t = []
recall_t = []
f1_t = []
auc_t = []
aupr_t = []
for repeat in range(repeats):
    train_loader, test_loader, valid_loader = get_train_test(train_data,valid_data,test_data,batch_size)
    #auc, precision, acc, f1, aupr = run_Adversarial_model(dataset, train_loader, test_loader,valid_loader,network_number)
    adversarial_model = Adversarial(in_dim=128, network_numbers=network_number)
    criterion = nn.CrossEntropyLoss()
    best_valid_dir = train_Adversarial_Model(dataset, train_loader, valid_loader, adversarial_model, criterion,initial_learning_rate)
    model_directory = 'output1/'+dataset+'_model'

    model_files = os.listdir(model_directory)
    aucs = []
    precisions = []
    accs = []
    f1s = []
    auprs = []

    for model_file in model_files:
        model_path = os.path.join(model_directory, model_file)
        auc, precision, acc, f1, aupr = test_Adversarial_Model(test_loader, adversarial_model, model_path)
#        write_infor='ROC-AUC:{:.4f},  Accuracy:{:.4f},Precision:{:.4f}, F1_score:{:.4f}\n'.format(aupr,acc, precision, f1)
#            infor = 'repeat:{}, ROC-AUC:{:.4f}, Precision:{:.4f}, Accuracy:{:.4f}, F1_score:{:.4f}, AUPR:{:.4f}\n'.format(
    #    repeat + 1, acc, precision, f1, auc, aupr)
#        print(write_infor)

        aucs.append(auc)
        precisions.append(precision)
        accs.append(acc)
        f1s.append(f1)
        auprs.append(aupr)
    max_auc_index = aucs.index(max(aucs))
    # print(f"Highest AUC: {aucs[max_auc_index]} from model: {model_files[max_auc_index]}")
    max_aupr_index = auprs.index(max(auprs))
    # print(f"Highest AUC: {aucs[max_aupr_index]} from model: {model_files[max_aupr_index]}")

#    print(f"Highest auc2: {auprs[max_aupr_index]} from model: {model_files[max_aupr_index]},acc{accs[max_aupr_index]},precisions{precisions[max_aupr_index]},f1s:{f1s[max_aupr_index]}")


    acc_t.append(acc)
    precision_t.append(precision)
    f1_t.append(f1)
    auc_t.append(auc)
    aupr_t.append(aupr)
    write_infor = 'repeat:{}, ROC-AUC:{:.4f}, Precision:{:.4f}, Accuracy:{:.4f}, F1_score:{:.4f}, AUPR:{:.4f}\n'.format(
        repeat + 1,auc,  precision,acc, f1,  aupr)
    print(write_infor)
    outfile.write(write_infor)