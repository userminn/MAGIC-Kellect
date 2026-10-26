import torch
import warnings
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata, transform_graph
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
import csv
import os
from sklearn.neighbors import NearestNeighbors
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.config import build_args
warnings.filterwarnings('ignore')


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device)
    dataset_name = main_args.dataset
    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    elif dataset_name == 'kellect':
        main_args.num_hidden = 64
        main_args.num_layers = 3
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3
    set_random_seed(0)

    if dataset_name in ['streamspot', 'wget', 'kellect']:
        dataset = load_batch_level_dataset(dataset_name, main_args.window_size, main_args.max_windows)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        pooler = Pooling(main_args.pooling)
        if dataset_name == 'kellect':
            model.eval()
            graphs = dataset['dataset']
            split = max(2, int(len(graphs) * 0.7))
            embeddings = []
            window_meta = graphs.window_metadata
            with torch.no_grad():
                for graph, _ in graphs:
                    graph = transform_graph(graph, main_args.n_dim, main_args.e_dim).to(device)
                    embeddings.append(pooler(graph, model.embed(graph)).cpu().numpy().reshape(-1))
            embeddings = np.asarray(embeddings)
            train_x = embeddings[:split]
            test_x = embeddings[split:]
            mean = train_x.mean(axis=0)
            std = train_x.std(axis=0) + 1e-6
            train_x = (train_x - mean) / std
            test_x = (test_x - mean) / std
            neighbors = min(3, len(train_x) - 1)
            train_nn = NearestNeighbors(n_neighbors=neighbors + 1).fit(train_x)
            train_dist, _ = train_nn.kneighbors(train_x)
            baseline = train_dist[:, 1:].mean(axis=1)
            threshold = float(np.percentile(baseline, 95))
            test_nn = NearestNeighbors(n_neighbors=neighbors).fit(train_x)
            test_dist, _ = test_nn.kneighbors(test_x)
            scores = test_dist.mean(axis=1)
            output_path = './eval_result/kellect_detection.csv'
            os.makedirs('./eval_result', exist_ok=True)
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as output:
                writer = csv.writer(output)
                writer.writerow(['window', 'first_timestamp', 'last_timestamp', 'event_count',
                                 'knn_distance', 'threshold_95pct', 'anomaly'])
                for i, score in enumerate(scores, split):
                    meta = window_meta[i]
                    writer.writerow([i, meta['first_timestamp'], meta['last_timestamp'],
                                     meta['event_count'], '{:.8f}'.format(score),
                                     '{:.8f}'.format(threshold), int(score > threshold)])
            print('Kellect windows: {} train, {} scored'.format(split, len(scores)))
            print('Training 95th-percentile kNN threshold: {:.6f}'.format(threshold))
            print('Flagged windows: {} / {}'.format(int(np.sum(scores > threshold)), len(scores)))
            print('Window scores saved to {}'.format(output_path))
            test_auc, test_std = float('nan'), 0.0
        else:
            test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], args.dataset, main_args.n_dim,
                                                        main_args.e_dim)
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        model.eval()
        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g).cpu().numpy())
                del g
            x_train = np.concatenate(x_train, axis=0)
            skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                x_test.append(model.embed(g).cpu().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)

            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            malicious_dict = {}
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            # Exclude training samples from the test set
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test
            test_auc, test_std, _, _ = evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test,
                                                                       result_y_test)
    if dataset_name != 'kellect':
        print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
    return


if __name__ == '__main__':
    args = build_args()
    main(args)
