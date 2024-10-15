import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
import numpy as np
import copy
import networkx as nx
from ganea import Autoencoder
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler
def orthogonal_projection(v1, v2):
    # 计算投影方向
    u = v1 / torch.norm(v1)

    # 计算在投影方向上的投影分量
    projection_v2 = torch.sum(v2 * u) * u

    # 计算与投影方向正交的分量
    orthogonal_component = v2 - projection_v2

    return orthogonal_component

# 定义EvolveGCN模型
class EvolveGCN(nn.Module):
    def __init__(self, in_feats, out_feats, num_time_steps):
        super(EvolveGCN, self).__init__()
        self.num_time_steps = num_time_steps
        self.gcn_layers = nn.ModuleList(
            [GraphConv(in_feats, out_feats, allow_zero_in_degree=True) for _ in range(num_time_steps)])

    def forward(self, g_list, features):
        diff_features = []
        for i in range(self.num_time_steps):
            g = g_list[i]
            if i < 1:
                # print(features[i].shape,type(features[i]))
                features[i] = self.gcn_layers[i](g, features[i])
                # print(features[i].shape)
                diff_features.append(features[i])
            else:
                # print("features[i-1]:",features[i-1])
                # print("features[i][0]:", features[i][0])
                features[i][:g_list[i - 1].number_of_nodes()] = features[i - 1]
                features[i] = self.gcn_layers[i](g, features[i])
                diff = features[i].clone()
                for j in np.arange(g_list[i - 1].number_of_nodes()):
                    # print("features[i][j].shape:", features[i][j].shape)
                    diff[j] = orthogonal_projection(features[i - 1][j], features[i][j])
                diff_features.append(diff)
        return features, diff_features


# # 生成示例图列表，模拟不同时间步的图演化
# def generate_graphs(num_nodes, num_time_steps):
#     graphs = []
#     for t in range(num_time_steps):
#         # 使用新的建议方式创建图
#         g = dgl.DGLGraph()
#         g.add_nodes(num_nodes[t])
#         src_nodes = np.arange(num_nodes[t])
#         dest_nodes = np.roll(src_nodes, shift=1)  # 修正连接方式
#         g.add_edges(src_nodes, dest_nodes)
#         graphs.append(g)
#
#     return graphs


import pandas as pd
from enhance import Generator, Discriminator, train_gan, generate_anomaly_features
from sklearn.model_selection import train_test_split


def set_new_column(row, values):
    for value in values:
        if row[2] > value and row['B'] < value:
            return 0
    return 1


def add_values_to_list(row, lst):
    if row[f'DF_snapshot_{time_id}'] == 1:
        lst.append((row[0], row[1]))


def getgraph(first_list_left_node, first_list_right_node):
    node = copy.deepcopy(first_list_left_node)
    node.extend(first_list_right_node)
    all_node = list(set(node))
    # all_nodes =[i for i in np.arange(max(first_node))]
    all_edges = []
    for i in np.arange(len(first_list_left_node)):
        all_edges.append([int(first_list_left_node[i]), int(first_list_right_node[i])])
    graph = nx.Graph()
    for nodei in all_node:
        graph.add_node(nodei)
    for edge in all_edges:
        graph.add_edge(edge[0], edge[1])
    dglgraph = dgl.from_networkx(graph)
    return dglgraph


def select_Node_emebdding(G, node_features, first_node, first_ano):
    ano = []
    nodeset = []
    featureset = []
    first_node = list(set(first_node))
    # print("节点特征与节点对应关系：")
    for node in G.nodes():
        # 获取节点特征
        features = node_features[node]
        if node in first_node:
            # print(f"节点 {node} 的特征：{features[0]}")
            nodeset.append(node)
            featureset.append(features.tolist())
            if node in first_ano:
                ano.append(1)
            else:
                ano.append(0)
    # print(len(nodeset), "nodeset", len(ano))

    # 在这里你可以对节点和特征进行操作
    # print("Node:", node)
    # print("Features:", features)

    # for node, features in zip(G.nodes, node_features):
    #     print("node",node)
    #     if node in first_node:
    #         # print(f"节点 {node} 的特征：{features[0]}")
    #         nodeset.append(node)
    #         featureset.append(features.tolist())
    #         if node in first_ano:
    #             ano.append(1)
    #         else:
    #             ano.append(0)
    # print(len(nodeset),"nodeset",len(ano))
    return nodeset, featureset, ano


def original_ano(first_ano_label, first_featureset):
    zero_indices = np.where(first_ano_label == 1)[0]
    # 取异常的index
    first_original_ano_features = first_featureset[zero_indices]
    # first_original_ano_features=np.squeeze(first_original_ano_features, axis=1)
    # print("first_featureset.shape:",first_featureset.shape)
    # print("type(first_original_ano_features)：",type(first_original_ano_features))
    return first_original_ano_features


def count_difference(lst):
    count_0 = np.sum(lst == 0)
    count_1 = np.sum(lst == 1)
    # count_0 = lst.count(0)  # 计算0的个数
    # count_1 = lst.count(1)  # 计算1的个
    difference = abs(count_0 - count_1)  # 计算差值
    return difference


# 主函数
if __name__ == '__main__':


    in_feats = 128
    out_feats = 128
    encoding_dim_autoE = 32
    autoencoder = Autoencoder(in_feats, encoding_dim_autoE)
    dataname = "mooc"
    data = pd.read_csv("data/" + dataname + ".csv", header=None)
    # data=data[:500]
    df2 = data[3].value_counts()
    print(df2)
    start = min(data[2])  # 起始值
    end = max(data[2])  # 结束值
    n = 5  # 分成n等分点

    # 计算等分点的间隔
    interval = (end - start) / n

    # 生成等分点的列表
    time_points = [start + (i + 1) * interval for i in range(n)]
    time_ids = [i for i in np.arange(len(time_points))]

    print(time_points)

    for time_id in time_ids:
        new_column_name = f'DF_Column_{time_id}'
        data[new_column_name] = data[2].apply(lambda x: 1 if x < time_points[time_id] else 0)

        if time_id == 0:
            new_snapshot_name = f'DF_snapshot_{time_id}'
            data[new_snapshot_name] = data[2].apply(lambda x: 1 if x <= time_points[time_id] else 0)
        else:
            new_snapshot_name = f'DF_snapshot_{time_id}'
            data[new_snapshot_name] = data[2].apply(lambda x: 1 if (
                    time_points[time_id] >= x > time_points[time_id - 1]) else 0)

        new_ano_name = f'DF_ano_{time_id}'
        data[new_ano_name] = data.apply(lambda row: row[1] if (row[2] <= time_points[time_id] and row[3] == 1) else -1,
                                        axis=1)
    # pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列

    # data['first_ano'] = np.where((data[2] <= first) & (data[3] == 1), data[1], -1)

    print(data)

    left_node_lists = {time_id: [] for time_id in time_ids}
    right_node_lists = {time_id: [] for time_id in time_ids}
    ano_node_lists = {time_id: [] for time_id in time_ids}
    snapshot_lists = {time_id: [] for time_id in time_ids}
    # for item in left_node_lists:
    #     item=data['column_name'].tolist()
    graph_list = []
    for time_id in time_ids:
        condition = data[f'DF_Column_{time_id}'] == 1
        condition_ano = data[f'DF_ano_{time_id}'] != -1
        condition_snap = data[f'DF_snapshot_{time_id}'] == 1
        left_node_lists[time_id] = data.loc[condition, 0].tolist()
        # 每个时刻之前的左节点
        right_node_lists[time_id] = data.loc[condition, 1].tolist()
        # 右节点
        ano_node_lists[time_id] = data.loc[condition_ano, f'DF_ano_{time_id}'].tolist()
        # print("len(ano_node_lists[time_id]",len(ano_node_lists[time_id] ))
        # 每一个时刻以前的异常节点
        snapshot_lists[time_id] = data.loc[condition_snap, 0].tolist() + data.loc[condition_snap, 1].tolist()
        # print("len(snapshot_lists[time_id]):",len(snapshot_lists[time_id]),snapshot_lists[time_id])
        # 当前快照的节点
        graph = getgraph(left_node_lists[time_id], right_node_lists[time_id])
        graph_list.append(graph)

    num_nodes = [g.number_of_nodes() for g in graph_list]
    # print("num_nodes:", num_nodes)
    # 随机初始化节点特征
    features = [torch.randn(num, in_feats) for num in num_nodes]
    # 创建EvolveGCN模型
    model = EvolveGCN(in_feats, out_feats, n)
    output_features, diff_features = model(graph_list, features)

    # print(output_features[0].shape)
    # print(diff_features[0].shape)

    node_set_lists = {time_id: [] for time_id in time_ids}  # 原图中节点集
    feature_set_lists = {time_id: [] for time_id in time_ids}  # 原图中特征
    ano_label_lists = {time_id: [] for time_id in time_ids}  # 原图中是否异常的标签
    generate_label_lists = {time_id: [] for time_id in time_ids}  # 原图中是否生成的标签
    timestamp_label_lists = {time_id: [] for time_id in time_ids}  # 原图中timestamp标签
    original_ano_features_lists = {time_id: [] for time_id in time_ids}  # 原始的异常特征
    num_samples_lists = {time_id: [] for time_id in time_ids}  # 生成样本数量
    generated_features_lists = {time_id: [] for time_id in time_ids}  # 生成特征
    generated_ano_list = {time_id: [] for time_id in time_ids}  # 生成的异常标签
    generated_timestamp_lists = {time_id: [] for time_id in time_ids}  # 生成部分的timestamp标签
    feature_combined_lists = {time_id: [] for time_id in time_ids}  # 原始特征和生成特征合并

    generated_label_lists = {time_id: [] for time_id in time_ids}  # 生成部分的生成标签
    ano_all_label_lists = {time_id: [] for time_id in time_ids}  # 所有的异常标签
    timestamp_all_label_lists = {time_id: [] for time_id in time_ids}  # 所有的timestamp标签
    generated_all_label_lists = {time_id: [] for time_id in time_ids}  # 所有的生成标签

    timestamp_add_feature_lists = {time_id: [] for time_id in time_ids}  # timestamp+feature

    generate_add_timestamp_add_feature_lists = {time_id: [] for time_id in time_ids}  # generate+timestamp+feature

    for time_id in time_ids:

        print(graph_list[time_id])
        # print("len(snapshot_lists[time_id]):",len(snapshot_lists[time_id]))
        # print("len(ano_node_lists[time_id]):",len(ano_node_lists[time_id]))
        node_set_lists[time_id], feature_set_lists[time_id], ano_label_lists[time_id] = select_Node_emebdding(
            graph_list[time_id], output_features[time_id], snapshot_lists[time_id], ano_node_lists[time_id])
        #node_set_lists[time_id]当前snapshot节点集，feature_set_lists[time_id]对应特征集， ano_label_lists[time_id]标签集0，1
        # print("ano_label_lists[time_id]::::",type(ano_label_lists[time_id]),len(ano_label_lists[time_id]))
        generate_label_lists[time_id] = [0 for i in node_set_lists[time_id]]
        #初始化一个和节点集同样长度的生成标签集，原始数据为0
        timestamp_label_lists[time_id] = [time_id for i in node_set_lists[time_id]]
        #为节点集初始化timestamp标签集
        feature_set_lists[time_id] = np.array(feature_set_lists[time_id])
        ano_label_lists[time_id] = np.array(ano_label_lists[time_id])
        generate_label_lists[time_id] = np.array(generate_label_lists[time_id])
        timestamp_label_lists[time_id] = np.array(timestamp_label_lists[time_id])
        if time_id != time_ids[-1]:

            # print("ano_label_lists[time_id]111",type(ano_label_lists[time_id]),ano_label_lists[time_id].shape)
            # print("feature_set_lists[time_id]",type(feature_set_lists[time_id]),feature_set_lists[time_id].shape)

            original_ano_features_lists[time_id] = original_ano(ano_label_lists[time_id], feature_set_lists[time_id])
            if original_ano_features_lists[time_id].size==0:
                feature_combined_lists[time_id] = feature_set_lists[time_id]
                ano_all_label_lists[time_id] = np.array(ano_label_lists[time_id])
                timestamp_all_label_lists[time_id] = np.array(timestamp_label_lists[time_id])
                generated_all_label_lists[time_id] = np.array(generate_label_lists[time_id])
            else:
                print("original_ano_features_lists[time_id]",original_ano_features_lists[time_id].shape,type(original_ano_features_lists[time_id]))
                num_samples_lists[time_id] = count_difference(ano_label_lists[time_id])
                autoencoder.train(original_ano_features_lists[time_id], num_epochs=200, learning_rate=0.001)
                generated_features_lists[time_id] = autoencoder.generate_abnormal_features(original_ano_features_lists[time_id],num_samples_lists[time_id])

                # generator = train_gan(original_ano_features_lists[time_id], num_epochs=200)  # 增加训练周期

                # generated_features_lists[time_id] = generate_anomaly_features(generator, latent_dim=128,
                #                                                               num_samples=num_samples_lists[time_id])

                feature_combined_lists[time_id] = np.vstack(
                    (feature_set_lists[time_id], generated_features_lists[time_id]))

                # 将原特征与生成特征合并
                generated_ano_list[time_id] = [1] * generated_features_lists[time_id].shape[0]
                # 生成部分初始化异常标签
                generated_ano_list[time_id] = np.array(generated_ano_list[time_id])

                generated_label_lists[time_id] = [1] * generated_features_lists[time_id].shape[0]
                # 为生成部分初始化生成标签
                generated_label_lists[time_id] = np.array(generated_label_lists[time_id])

                generated_timestamp_lists[time_id] = [time_id] * generated_features_lists[time_id].shape[0]
                # 生成部分的timestamp标签

                generated_timestamp_lists[time_id] = np.array(generated_timestamp_lists[time_id])

                ano_all_label_lists[time_id] = np.concatenate((ano_label_lists[time_id], generated_ano_list[time_id]))
                # 完整异常标签
                timestamp_all_label_lists[time_id] = np.concatenate(
                    (timestamp_label_lists[time_id], generated_timestamp_lists[time_id]))
                # 完整的timestamp标签
                generated_all_label_lists[time_id] = np.concatenate(
                    (generate_label_lists[time_id], generated_label_lists[time_id]))
                # 完整的是否生成的标签


        else:
            #最后一层用于测试集不用注入异常
            feature_combined_lists[time_id] = feature_set_lists[time_id]
            ano_all_label_lists[time_id] = np.array(ano_label_lists[time_id])
            timestamp_all_label_lists[time_id] = np.array(timestamp_label_lists[time_id])
            generated_all_label_lists[time_id] = np.array(generate_label_lists[time_id])
            # 最后一层上不注入异常

        timestamp_all_label_lists[time_id] = timestamp_all_label_lists[time_id][:, np.newaxis]
        generated_all_label_lists[time_id] = generated_all_label_lists[time_id][:, np.newaxis]
        #(10,)-(10,1)

        timestamp_add_feature_lists[time_id] = np.concatenate(
            [timestamp_all_label_lists[time_id], feature_combined_lists[time_id]], axis=1)
        #[timestamp_label, feature]
        generate_add_timestamp_add_feature_lists[time_id] = np.concatenate(
            [generated_all_label_lists[time_id], np.array(timestamp_add_feature_lists[time_id])], axis=1)
        #列[generate_label,timestamp_label,feature]


print("generate_add_timestamp_add_feature_lists",type(generate_add_timestamp_add_feature_lists))
my_list = []
for time_id, feature in generate_add_timestamp_add_feature_lists.items():
    my_list.append(feature)
train_feature = np.vstack(my_list[:-1])
#转换成列表然后把列表中除最后用于测试的一项，其他项都纵向合并
my_ano_all_label_list = []
for time_id, feature in ano_all_label_lists.items():
    print("feature",type(feature),feature.shape)
    feature=feature[:, np.newaxis]
    #（10，）-（10，1）
    print("feature",type(feature),feature.shape)
    my_ano_all_label_list.append(feature)
train_ano_label = np.vstack(my_ano_all_label_list[:-1])
#转换成列表然后把列表中除最后用于测试的一项，其他项都纵向合并


# train_ano_label = np.vstack(ano_all_label_lists[:-1])
last_snapshot_feature = generate_add_timestamp_add_feature_lists[time_ids[-1]]
last_snapshot_ano_label = ano_all_label_lists[time_ids[-1]]
#取用于测试的项
rus = RandomUnderSampler(random_state=0, replacement=True)
enn = EditedNearestNeighbours()
#欠采样

train_last_infor, test_infor, train_last_label, test_label = train_test_split(
    last_snapshot_feature, last_snapshot_ano_label,
    test_size=1 - 0.1)
 #对用于测试的项，划分一部分作为测试集

train_feature = np.concatenate([train_feature, train_last_infor], axis=0)
train_last_label=train_last_label[:,np.newaxis]
train_ano_label = np.concatenate([train_ano_label, train_last_label], axis=0)
#剩余部分融入训练部分整体
train_infor, valid_infor, train_label, valid_label = train_test_split(train_feature, train_ano_label,
                                                                      test_size=0.2)
#划分训练集，验证集
#train,valid,test，写入文件

# with open("processed_data/" + dataname + 'all.txt', 'w') as f:
#     for i in np.arange(len(train_label)):
#         # print("train_label[i]", type(train_label[i]), train_label[i].shape)
#         label = str(int(train_label[i][0]))
#         embedding = ','.join([str(x) for x in train_infor[i]])
#         f.write(f"{label},{embedding}\n")
#     for i in np.arange(len(valid_label)):
#         label = str(int(valid_label[i][0]))
#         embedding = ','.join([str(x) for x in valid_infor[i]])
#         f.write(f"{label},{embedding}\n")
#     for i in np.arange(len(test_label)):
#         # print("test_label[i]", type(test_label[i]), test_label[i].shape)
#         label = str(int(test_label[i]))
#         embedding = ','.join([str(x) for x in test_infor[i]])
#         f.write(f"{label},{embedding}\n")

print("embedding finish!")

with open("data_AE/"+dataname + 'train.txt', 'w') as f:
    for i in np.arange(len(train_label)):
        print("train_label[i]", type(train_label[i]), train_label[i])
        label = str(int(train_label[i]))
        embedding = ','.join([str(x) for x in train_infor[i]])
        f.write(f"{label},{embedding}\n")
        # label,is_generate,timestamp,embedding
with open("data_AE/"+dataname + 'valid.txt', 'w') as f:
    for i in np.arange(len(valid_label)):
        label = str(int(valid_label[i]))
        embedding = ','.join([str(x) for x in valid_infor[i]])
        f.write(f"{label},{embedding}\n")
with open("data_AE/"+dataname + 'test.txt', 'w') as f:
    for i in np.arange(len(test_label)):
        label = str(int(test_label[i]))
        embedding = ','.join([str(x) for x in test_infor[i]])
        f.write(f"{label},{embedding}\n")

        # in_feats = 32
    # out_feats = 32
    #
    #
    #
    # # 定义动态网络数据，每个时间步对应一个图
    # G1 = dgl.graph(([0, 1, 3], [1, 3,0]))
    # G2 = dgl.graph(([0, 1, 2, 4], [4, 2, 3, 0]))
    # G3 = dgl.graph(([0, 1, 2,5,4], [1, 2, 0,3,5]))
    #
    # # 遍历所有节点
    #
    #
    #
    # # 将图放入一个列表中，这个列表就是完整的 graph_list
    # graph_list = [G1, G2, G3]
    # num_time_steps = int(len(graph_list))
    # # 生成示例图列表
    # # graphs = generate_graphs(num_nodes, num_time_steps)
    # # print("graphs",len(graphs))
    # num_nodes = [G1.number_of_nodes(),G2.number_of_nodes(),G3.number_of_nodes()]
    #
    # # 随机初始化节点特征
    # features = [torch.randn(num, in_feats)for num in num_nodes]

    #
    # # 运行模型
    # output_features = model(graph_list, features)
    #
    # print(output_features[0].shape)
    # for t in range(num_time_steps):
    #     for node,feature in zip(graph_list[t].nodes().numpy(),output_features[t]):
    #     # print("output_features[t]:",output_features[t])
    #         print("",t,"",node,feature)
