from utils import universal_load_data, Data
from sklearn.utils import shuffle
import torch
import numpy as np


train_fts_ratio = 0.4*0.1

from torch_geometric.utils import degree

dataset_name = 'citeseer'


def analyze_basics(adj, features):
    """Анализ базовых характеристик графа"""
    num_nodes = features.size(0)
    num_edges = adj.size(1) // 2  # Для ненаправленного графа

    return {
        "num_nodes": torch.tensor(num_nodes),
        "num_edges": torch.tensor(num_edges),
        "density": torch.tensor(num_edges / (num_nodes * (num_nodes - 1))),
        "is_directed": torch.tensor(False)  # Так как загружаем ненаправленные графы
    }


def degree_analysis(edge_index, num_nodes):
    """Анализ распределения степеней через edge_index"""
    deg = degree(edge_index[0], num_nodes=num_nodes)
    return {
        "max_degree": deg.max(),
        "min_degree": deg.min(),
        "median_degree": deg.median(),
        "degree_std": deg.float().std()
    }


# def clustering_coefficient(edge_index, num_nodes):
#     """Расчет коэффициента кластеризации для sparse графа"""
#     from torch_cluster import triangle_count
#     triangles = triangle_count(edge_index, num_nodes=num_nodes)
#     deg = degree(edge_index[0], num_nodes=num_nodes)
#     possible_triads = deg * (deg - 1)
#     coefficients = triangles.float() / possible_triads.float().clamp(min=1)
#     return {"avg_clustering": coefficients[~coefficients.isnan()].mean()}


def connected_components(edge_index, num_nodes):
    """Поиск компонент связности через BFS"""
    from collections import deque
    adj_list = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    visited = torch.zeros(num_nodes, dtype=torch.bool)
    components = []

    for node in range(num_nodes):
        if not visited[node]:
            queue = deque([node])
            visited[node] = True
            component = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adj_list[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(torch.tensor(component))

    component_sizes = torch.tensor([len(c) for c in components])
    return {
        "num_components": torch.tensor(len(components)),
        "largest_component_size": component_sizes.max(),
        "component_size_distribution": component_sizes.float() / component_sizes.sum()
    }


def feature_analysis(features):
    """Анализ характеристик узлов"""
    return {
        "feature_mean": features.mean(dim=0),
        "feature_std": features.std(dim=0),
        "nonzero_feature_ratio": (features != 0).float().mean(dim=1)
    }


def label_analysis(labels, idx_train, idx_val, idx_test):
    """Анализ распределения меток"""
    results = {}
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        if idx is not None:
            class_counts = torch.bincount(labels[idx])
            results[f"{name}_class_dist"] = class_counts.float() / class_counts.sum()
    return results


def homophily_analysis(edge_index, labels):
    """Расчет гомофилии через edge_index"""
    src, dst = edge_index
    same_labels = (labels[src] == labels[dst]).float().mean()
    return {"label_homophily": same_labels}

data_dict = universal_load_data(dataset_name, return_dict=True, norm_adj=False)

# Извлечение edge_index из sparse матрицы
edge_index = data_dict['adj'].coalesce().indices()

# Запуск анализов
analysis = {}
analysis.update(analyze_basics(data_dict['adj'], data_dict['features']))
analysis.update(degree_analysis(edge_index, data_dict['features'].size(0)))
# analysis.update(clustering_coefficient(edge_index, data_dict['features'].size(0)))
analysis.update(connected_components(edge_index, data_dict['features'].size(0)))
analysis.update(feature_analysis(data_dict['features']))
analysis.update(label_analysis(data_dict['labels'],
                data_dict['idx_train'],
                data_dict['idx_val'],
                data_dict['idx_test']))
analysis.update(homophily_analysis(edge_index, data_dict['labels']))

# Вывод результатов (используйте функцию print_analysis из предыдущего ответа)

def print_analysis(results):
    """Форматированный вывод результатов анализа графа"""

    # Базовые характеристики
    print("═" * 50)
    print(f"BASIC GRAPH INFO:")
    print(f"Nodes: {results['num_nodes'].item():,}")
    print(f"Edges: {results['num_edges'].item():,}")
    # print(f"Avg Degree: {results['avg_degree'].item():.2f}")
    print(f"Density: {results['density'].item():.4f}")
    print(f"Directed: {'Yes' if results['is_directed'] else 'No'}")

    # Степени узлов
    print("\n" + "═" * 50)
    print("DEGREE ANALYSIS:")
    print(f"Max degree: {results['max_degree'].item()}")
    print(f"Min degree: {results['min_degree'].item()}")
    print(f"Median degree: {results['median_degree'].item()}")
    print(f"Degree std: {results['degree_std'].item():.2f}")

    # Компоненты связности
    print("\n" + "═" * 50)
    print("CONNECTIVITY:")
    print(f"Connected components: {results['num_components'].item()}")
    print(f"Largest component: {results['largest_component_size'].item()} nodes")
    comp_sizes = results['component_size_distribution'].numpy()
    print(f"Component size distribution: {comp_sizes.round(3)}")

    # # Кластеризация
    # print("\n" + "═" * 50)
    # print("CLUSTERING:")
    # print(f"Avg. clustering coeff: {results['avg_clustering'].item():.3f}")

    # Особенности узлов
    print("\n" + "═" * 50)
    print("FEATURE STATISTICS:")
    print(f"Global feature mean: {results['feature_mean'].mean().item():.10f}")
    print(f"Global feature std: {results['feature_std'].mean().item():.10f}")
    print(f"Avg. non-zero features per node: {results['nonzero_feature_ratio'].mean().item():.2%}")

    # Метки и гомофилия
    print("\n" + "═" * 50)
    print("LABEL ANALYSIS:")
    print(f"Label homophily: {results['label_homophily'].item():.3f}")

    # Распределение меток
    for split in ['train', 'val', 'test']:
        dist = results[f"{split}_class_dist"].numpy()
        print(f"\n{split.upper()} class distribution:")
        for i, p in enumerate(dist):
            print(f"  Class {i}: {p:.2%}", end="  ")
        print()

# Запуск анализа и вывод

print_analysis(analysis)
