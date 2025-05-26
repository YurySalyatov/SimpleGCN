import os

import torch
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import csv
import statistics

from utils import universal_load_data, Data
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Загрузка данных
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0]
# print(data.x.shape)
# print(data.edge_index.shape)

# dataset_str = 'citeseer'


def node_noise(data, percentage):
    """
    Заменяет все фичи у случайно выбранного процента вершин на значения из общего распределения тензора
    Args:
        tensor: исходный тензор (num_nodes, num_features)
        percentage: процент вершин для замены (0.0 - 1.0)
    Returns:
        тензор с шумом
    """
    noisy_data = data.clone()
    noisy_data.to(device)
    tensor = noisy_data.x
    if percentage <= 0:
        return noisy_data

    num_nodes = tensor.size(0)
    num_selected = int(percentage * num_nodes)

    if num_selected == 0:
        return noisy_data

    # Выбираем случайные вершины
    selected_nodes = torch.randperm(num_nodes, device=device)[:num_selected]

    # Генерируем значения для замены из общего распределения
    flattened = tensor.flatten()
    shuffled_values = flattened[torch.randperm(len(flattened), device=device)][:num_selected * tensor.size(1)]
    replacement = shuffled_values.view(num_selected, tensor.size(1))

    # Создаем копию и применяем шум
    noised_tensor = tensor.clone()
    noised_tensor[selected_nodes] = replacement
    noisy_data.x = noised_tensor
    return noisy_data


def feature_noise(data, percentage):
    noisy_data = data.clone()
    noisy_data.to(device)
    tensor = noisy_data.x
    if percentage <= 0:
        return noisy_data

    num_features = tensor.size(1)
    # print("num_features", num_features)
    num_selected_features = int(percentage * num_features)
    # print("num_selected_features", num_selected_features)
    if num_selected_features == 0:
        return noisy_data

    # Выбираем случайные фичи
    selected_features = torch.randperm(num_features, device=device)[:num_selected_features]
    # print("selected_features", selected_features)
    # Генерируем значения для замены
    flattened = tensor.flatten()
    shuffled_values = flattened[torch.randperm(len(flattened), device=device)][:tensor.size(0) * num_selected_features]
    replacement = shuffled_values.view(tensor.size(0), num_selected_features)

    # Создаем копию и применяем шум
    noised_tensor = tensor.clone()
    # print("noisy tensor")
    # print(noised_tensor[:, selected_features])
    noised_tensor[:, selected_features] = replacement
    # print(noised_tensor[:, selected_features])
    noisy_data.x = noised_tensor
    return noisy_data


# Модель GCN, GAT, SAGE
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16, dropout=0.5, layer_name="GCN", heads=4):
        super().__init__()
        self.layer_name = layer_name
        self.dropout = dropout

        if layer_name == "GCN":
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, num_classes)
        elif layer_name == "GAT":
            self.conv1 = GATConv(num_features, hidden_dim, heads=heads)
            self.conv2 = GATConv(hidden_dim * heads, num_classes)
        elif layer_name == "SAGE":
            self.conv1 = SAGEConv(num_features, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, num_classes)
        else:
            raise Exception(f"Unknown layer name: {layer_name}, expected on of GCN, GAT, SAGE")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        if self.layer_name == "GAT":
            x = F.elu(x)
        else:
            x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def save_table(results, filename="results_experiment1_table.txt"):
    all_keys = [k for k in results[0].keys() if k != 'sigma']
    # Создаем список форматов: первый столбец .2f, остальные .5f
    headers = ["Noise Level"] + all_keys
    float_fmts = [".2f"] + [".5f"] * (len(headers) - 1)
    data = [[res[key] for res in results] for key in results[0].keys()]
    rows = list(zip(*data))
    # Генерируем таблицу
    table_str = tabulate(
        rows,
        headers=headers,
        tablefmt="grid",
        floatfmt=float_fmts
    )

    # Сохраняем в файл
    with open(filename, "w") as f:
        f.write(table_str)

    return table_str


def plot_all_results(results, save_path=None):
    # Получаем все ключи для графиков (исключая sigma)
    plot_keys = [k for k in results[0].keys() if k != 'sigma']

    for key in plot_keys:
        plt.figure(figsize=(10, 6))
        plt.plot(
            [res['sigma'] for res in results],
            [res[key] for res in results],
            marker='o',
            label=key
        )
        plt.xlabel('Noise Level')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'Dependence of {key} on Noise Level')
        plt.grid(True)

        if save_path:
            # Заменяем пробелы в названии ключа для имени файла
            filename = f"{key.replace(' ', '_')}_plot.png"
            plt.savefig(f"{save_path}/{filename}")
            plt.close()
        plt.show()


def calculate_metrics_spread(results, print_values=True):
    """
    Рассчитывает разброс значений для каждой метрики (мин, макс, среднее, стандартное отклонение, размах).
    Исключает ключ 'sigma', так как это параметр, а не метрика.

    Args:
        data (list of dict): Список словарей с данными.

    Returns:
        dict: Словарь с разбросом для каждой метрики.
    """
    # Собираем все значения для каждого ключа (кроме 'sigma')
    metric_values = {}
    for entry in results:
        for key, value in entry.items():
            if not key.endswith("PU"):
                continue  # Пропускаем параметр не PU
            if key not in metric_values:
                metric_values[key] = []
            metric_values[key].append(
                float(value))  # Предполагаем, что значения можно преобразовать в float

    # Рассчитываем статистики для каждой метрики
    result = {}
    for metric, values in metric_values.items():
        if len(values) < 2:
            continue  # Недостаточно данных для расчёта

        result[metric] = {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'stdev': statistics.stdev(values),
            'range': max(values) - min(values),
        }
    if print_values:
        stats_table = []
        for metric, values in metric_values.items():
            stats_table.append({
                "Metric": metric,
                "Min": min(values),
                "Max": max(values),
                "Mean": statistics.mean(values),
                "Stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "Range": max(values) - min(values)
            })
        print("Статистики по метрикам:")
        print(tabulate(stats_table,
                       headers="keys",
                       floatfmt=".4f",
                       tablefmt="grid"))
    return result


# Обучение модели
def train_model(model, data, dataset_name, layer, epochs=4000, target_acc=0.8):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    loss_f = torch.nn.CrossEntropyLoss()
    min_loss = 1e10
    max_acc = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_f(out[data.train_idx], data.y[data.train_idx])
        pred = out.argmax(dim=1)
        val_acc = (pred[data.val_idx] == data.y[data.val_idx]).sum() / data.val_idx.shape[0]
        if max_acc < val_acc:
            torch.save(model.state_dict(), f"output/best_GCN_model_{dataset_name}_{layer}.pkl")
            min_loss = min(min_loss, loss)
            max_acc = val_acc
            if target_acc <= max_acc:
                break
        # if epoch % 100 == 0:
        # print(f"loss: {loss.item():.4f}, epoch: {epoch + 1}")
        loss.backward()
        optimizer.step()
    print(f"min loss: {min_loss:.4f}")
    print(f"max_acc: {max_acc}")
    return model, max_acc, min_loss


# Вычисление энтропии
def compute_entropy(log_probs):
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=1)


# def compute_entropy_exp(log_probs):
#     return -torch.sum(torch.exp(log_probs) * log_probs, dim=1)


def get_normalize_std(array):
    mn = torch.mean(array, dim=0)
    std = torch.std(array, dim=0)
    return std / mn * 100


def compute_margin(log_probs):
    exp = torch.exp(log_probs)
    exp, _ = torch.sort(exp, dim=1, descending=True)
    return exp[:, 0] - exp[:, 1]


# Уровни шума и результаты
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def get_data(dataset_str):
    adj, features, labels, _, _, _ = universal_load_data(dataset_str)
    # print(features[:20, :20])
    shuffled_nodes = shuffle(np.arange(adj.shape[0]), random_state=seed)
    train_mask = torch.LongTensor(shuffled_nodes[:int(0.4 * adj.shape[0])])
    val_mask = torch.LongTensor(
        shuffled_nodes[int(0.4 * adj.shape[0]):int((0.4 + 0.1) * adj.shape[0])])
    test_mask = torch.LongTensor(shuffled_nodes[int((0.4 + 0.1) * adj.shape[0]):])
    data = Data(x=features, edge_index=adj, y=labels, train_idx=train_mask, val_idx=val_mask,
                test_idx=test_mask)
    num_features = features.shape[1]
    num_classes = torch.max(labels) + 1
    data.to(device)
    return data, num_features, num_classes


noisy_methods = [feature_noise]
datasets = ['cora', 'citeseer']
# Experiment 1
layers = ["GCN", "GAT", "SAGE"]
for dataset_name in datasets:
    os.makedirs(f"results/{dataset_name}", exist_ok=True)
    pred_data, num_features, num_classes = get_data(dataset_name)
    print(dataset_name)
    for method in noisy_methods:
        print(method.__name__)
        results = []
        for sigma in noise_levels:
            one_result = {"sigma": sigma}
            for layer in layers:
                print(layer)
                pu_arr = []
                acc_arr = []
                for _ in range(5):
                    data = method(pred_data, sigma)
                    model = GCN(num_features, num_classes, layer_name=layer)
                    # print(model)
                    model.to(device)
                    model, max_acc, _ = train_model(model, data, dataset_name, layer=layer,
                                                    target_acc=0.5 + 0.25 * (1 - sigma))
                    model.load_state_dict(torch.load(f"output/best_GCN_model_{dataset_name}_{layer}.pkl"))

                    # Оценка PU
                    num_samples = 20
                    predictions = []
                    model.train()
                    for _ in range(num_samples):
                        with torch.no_grad():
                            log_probs = model(data)
                            predictions.append(torch.exp(log_probs[data.test_idx]))

                    predictions = torch.stack(predictions)
                    mean_pred = predictions.mean(dim=0)
                    mean_pred_entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-18), dim=1)
                    pu = mean_pred_entropy.mean()
                    pu_arr.append(pu)
                    acc_arr.append(max_acc)
                pu_arr = torch.stack(pu_arr)
                acc_arr = torch.stack(acc_arr)
                mean_pu = pu_arr.mean().item()
                var_pu = pu_arr.var().item()
                one_result[f"{layer} PU"] = mean_pu
                one_result[f"{layer} var PU"] = var_pu
                one_result[f"{layer} max acc"] = acc_arr.max().item()
                one_result[f"{layer} min acc"] = acc_arr.min().item()
                one_result[f"{layer} mean acc"] = acc_arr.mean().item()
                one_result[f"{layer} var acc"] = acc_arr.var().item()
            print(one_result)
            results.append(one_result)
        # plot_dir = f"results/{dataset_name}/{method.__name__}/plots"
        # os.makedirs(plot_dir, exist_ok=True)
        table_file = f"results/{dataset_name}/{method.__name__}/table_experiment2.txt"
        os.makedirs(f"results/{dataset_name}/{method.__name__}", exist_ok=True)

        # plot_all_results(results, save_path=plot_dir)
        # save_table(results, filename=table_file)
