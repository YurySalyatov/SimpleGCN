import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from sklearn.utils import shuffle

from utils import universal_load_data, Data
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Классы моделей
class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.5):
        super().__init__()
        self.conv = GCNConv(num_features, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x = F.relu(self.conv(data.x, data.edge_index))
        return F.dropout(x, p=self.dropout, training=self.training)


class SimpleSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.5, normalize=False):
        super().__init__()
        self.conv = SAGEConv(num_features, num_classes, normalize=normalize)
        self.dropout = dropout

    def forward(self, data):
        x = F.relu(self.conv(data.x, data.edge_index))
        return F.dropout(x, p=self.dropout, training=self.training)


# Функция обучения
def train_model(model, data, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_idx], data.y[data.train_idx])
        loss.backward()
        optimizer.step()

        # Валидация
        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(dim=1)
            val_acc = accuracy_score(data.y[data.val_idx].cpu(),
                                     pred[data.val_idx].cpu())
            if val_acc > best_acc:
                best_acc = val_acc

    return best_acc


# Визуализация и тестирование
def analyze_models(data, num_features, num_classes):
    dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = {'GCN': [], 'SAGE_norm': [], 'SAGE_no_norm': []}
    best_drops = [0, 0, 0]
    # Тестируем GCN
    mx = 0
    for dropout in dropouts:
        model = SimpleGCN(num_features, num_classes, dropout=dropout)
        acc = train_model(model, data)
        if acc > mx:
            mx = acc
            best_drops[0] = dropout
        results['GCN'].append(acc)
        print(f'CGN, dropout:{dropout}, acc: {acc}')

    # Тестируем SAGE
    mx_n = 0
    mx_nn = 0
    for dropout in dropouts:
        # С нормализацией
        model = SimpleSAGE(num_features, num_classes, dropout=dropout, normalize=True)
        acc = train_model(model, data)
        if acc > mx_n:
            mx_n = acc
            best_drops[1] = dropout
        results['SAGE_norm'].append(acc)
        print(f'SAGE, normalize, dropout:{dropout}, acc: {acc}')
        # Без нормализации
        model = SimpleSAGE(num_features, num_classes, dropout=dropout, normalize=False)
        acc = train_model(model, data)
        if acc > mx_nn:
            mx_nn = acc
            best_drops[2] = dropout
        results['SAGE_no_norm'].append(acc)
        print(f'SAGE, not normalize, dropout:{dropout}, acc: {acc}')

    # Построение графиков
    plt.figure(figsize=(10, 6))

    # Цвета и стили линий
    styles = {
        'GCN': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'SAGE_norm': {'color': 'green', 'marker': 's', 'linestyle': '--'},
        'SAGE_no_norm': {'color': 'red', 'marker': 'd', 'linestyle': ':'}
    }

    for model_name in results:
        plt.plot(dropouts, results[model_name],
                 label=model_name,
                 **styles[model_name],
                 linewidth=2,
                 markersize=8)

    # Настройки графика
    plt.xlabel('Dropout Rate', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(dropouts)
    plt.ylim(0, 1.05)  # Фиксируем масштаб Y для всех
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

    # Аннотируем лучшие значения
    for model_name, values in results.items():
        best_acc = max(values)
        best_idx = values.index(best_acc)
        plt.annotate(f'{best_acc:.2f}',
                     (dropouts[best_idx], best_acc),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     color=styles[model_name]['color'])

    plt.tight_layout()
    plt.show()

    # T-SNE визуализация лучших моделей
    tsne_visualization(data, num_features, num_classes, best_drops)


def tsne_visualization(data, num_features, num_classes, bests_dropouts):
    models = {
        'Best GCN': SimpleGCN(num_features, num_classes, dropout=bests_dropouts[0]),
        'Best SAGE (norm)': SimpleSAGE(num_features, num_classes, dropout=bests_dropouts[1], normalize=True),
        'Best SAGE (no norm)': SimpleSAGE(num_features, num_classes, dropout=bests_dropouts[2], normalize=False)
    }

    plt.figure(figsize=(15, 4))
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            embeddings = model.conv(data.x, data.edge_index)

        tsne = TSNE(n_components=2)
        emb_2d = tsne.fit_transform(embeddings.cpu().numpy())

        plt.subplot(1, 3, i + 1)
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=data.y.cpu(), cmap='tab10', alpha=0.6)
        plt.title(f'{name} Embeddings')
        plt.colorbar()

    plt.tight_layout()
    plt.show()


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


data, num_features, num_classes = get_data('cora')

# Пример использования
analyze_models(data, num_features, num_classes)
