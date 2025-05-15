from simpleModel import SimpleGCNModel, SimpleGATModel
from torch.optim import Adam
import torch.nn.functional as F
from utils import universal_load_data, Data
import torch
import numpy as np
from sklearn.utils import shuffle
import os

train_fts_ratio = 0.4 * 1.0


def GCNTrain(model, data, epoch=1000, trace=True):
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    min_loss = None
    loss_arr = []
    best_epoch = 0
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # print(out.shape)
        train_loss = F.nll_loss(out[data.train_idx], data.y[data.train_idx])
        loss_arr.append(train_loss.item())
        valid_loss = F.nll_loss(out[data.valid_idx], data.y[data.valid_idx])
        if trace and i % 100 == 0:
            print(f"train loss: {train_loss.item():.4f}, valid loss: {valid_loss.item()} epoch: {i + 1}")
        if min_loss is None or valid_loss.item() < min_loss:
            min_loss = valid_loss.item()
            # torch.save(model.state_dict(), os.path.join(os.getcwd(), 'output',
            #                                             'best_GCN_Model.pkl'))
            best_epoch = epoch - 1
        train_loss.backward()
        optimizer.step()
    return best_epoch, min_loss, loss_arr, model.entropy, model.margin


adj, features, labels, _, _, _ = universal_load_data('cora')
norm_adj, _, _, _, _, _ = universal_load_data('cora', norm_adj=True)
shuffled_nodes = shuffle(np.arange(adj.shape[0]), random_state=42)
train_fts_idx = torch.LongTensor(shuffled_nodes[:int(train_fts_ratio * adj.shape[0])])
vali_fts_idx = torch.LongTensor(
    shuffled_nodes[int(0.4 * adj.shape[0]):int((0.4 + 0.1) * adj.shape[0])])
test_fts_idx = torch.LongTensor(shuffled_nodes[int((0.4 + 0.1) * adj.shape[0]):])

in_f = features.shape[1]
# print("labels", labels.detach().numpy())
out_f = len(set(labels.detach().numpy()))
print(features.shape)
print(adj.shape)
print(labels.shape)
# print(diag_fts.shape)
print(norm_adj.shape)
GCNModel = SimpleGCNModel(in_f, out_f)
GATModel = SimpleGATModel(in_f, out_f)
print(norm_adj.coalesce().indices().shape)
# edge_index = torch.stack(torch.where(norm_adj > 0), dim=0).long()
data = Data(x=features, edge_index=norm_adj, y=labels, train_idx=train_fts_idx, val_idx=vali_fts_idx,
            test_idx=test_fts_idx)
print(GCNModel)
results = GCNTrain(GCNModel, data)
