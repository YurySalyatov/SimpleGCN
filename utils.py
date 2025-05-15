import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx


def universal_load_data(
        dataset_str,
        norm_adj=True,
        add_self_loops_flag=True,
        return_dict=False,
        return_indices=True,
        generative_flag=False
):
    """
    Универсальная функция загрузки данных для GNN

    Параметры:
    dataset_str: str - название датасета (cora, citeseer, pubmed)
    norm_adj: bool - нормализовать матрицу смежности (default: True)
    add_self_loops_flag: bool - добавить self-loops (default: True)
    return_dict: bool - возвращать словарь вместо кортежа (default: False)
    return_indices: bool - возвращать train/val/test индексы (default: True)
    generative_flag: bool - не нормализовать фичи (default: False)

    Возвращает:
    Зависит от флагов:
    - По умолчанию: (adj, features, labels, idx_train, idx_val, idx_test)
    - При return_dict: словарь с полным набором данных
    """

    # Загрузка raw данных
    try:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for name in names:
            with open(f"./data/{dataset_str}/ind.{dataset_str}.{name}", 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = objects
    except:
        raise ValueError(f"Cannot load dataset: {dataset_str}")

    # Обработка индексов
    test_idx_reorder = parse_index_file(f"./data/{dataset_str}/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # Объединение фичей
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    # Обработка меток
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = torch.LongTensor(np.argmax(labels, axis=1))

    # Нормализация фичей
    if not generative_flag:
        features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))

    # Построение графа
    G = nx.from_dict_of_lists(graph)
    edge_list = adj_list_from_dict(graph)

    # Матрица смежности
    adj = nx.adjacency_matrix(G)
    if add_self_loops_flag:
        adj = adj + sp.eye(adj.shape[0])

    # Нормализация adjacency
    if norm_adj:
        adj = normalize_adj(adj)

    # Конвертация в тензоры
    adj = sparse_matrix_to_torch(adj)

    # Подготовка индексов
    if return_indices:
        idx_test = test_idx_range.tolist()
        idx_train = list(range(len(y)))
        idx_val = list(range(len(y), len(y) + 500))
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
    else:
        idx_train, idx_val, idx_test = None, None, None

    # Формирование результата
    if return_dict:
        return {
            'adj': adj,
            'features': features,
            'labels': labels,
            'edge_list': edge_list,
            'graph': G,
            'idx_train': idx_train,
            'idx_val': idx_val,
            'idx_test': idx_test
        }
    else:
        result = (adj, features, labels)
        if return_indices:
            result += (idx_train, idx_val, idx_test)
        return result


# Вспомогательные функции ----------------------------------------------------
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_features(features):
    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features)


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_matrix_to_torch(mat):
    mat = mat.tocoo()
    indices = torch.LongTensor(np.vstack((mat.row, mat.col)))
    values = torch.FloatTensor(mat.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(mat.shape))


def adj_list_from_dict(graph_dict):
    edge_list = []
    for node, neighbors in graph_dict.items():
        for neighbor in neighbors:
            edge_list.append((node, neighbor))
    return edge_list


def get_entropy(array):
    return -(array * np.log(array + 1e-7)).sum(axis=1)


def get_margin(array):
    sorted_matrix = np.sort(array, axis=1)  # Сортируем строки
    return sorted_matrix[:, -1] - sorted_matrix[:, -2]


class Data:
    def __init__(self, x, edge_index, y, train_idx, val_idx, test_idx):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

    def clone(self):
        return Data(self.x.detach().clone(),
                    self.edge_index.detach().clone(),
                    self.y.detach().clone(),
                    self.train_idx.detach().clone(),
                    self.val_idx.detach().clone(),
                    self.test_idx.detach().clone())

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        self.train_idx = self.train_idx.to(device)
        self.val_idx = self.val_idx.to(device)
        self.test_idx = self.test_idx.to(device)