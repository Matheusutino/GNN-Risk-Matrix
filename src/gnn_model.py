# -*- coding: utf-8 -*-
"""
GNNModel module for training and prediction using GraphSAGE with PyTorch Geometric.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import numpy as np
from tqdm import tqdm


class GraphSAGE(torch.nn.Module):
    """GraphSAGE model implementation."""

    def __init__(self, in_channels, hidden_channels=32, out_channels=2):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class GNNModel:
    """
    Wraps the PyTorch Geometric GraphSAGE model for training and prediction.
    """

    def __init__(self, hidden_channels=32):
        """
        Initializes the GNN model.

        Args:
            hidden_channels (int): Number of hidden channels in GraphSAGE layers.
        """
        self.model = None
        self.data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_channels = hidden_channels
        self.node_id_to_idx = {}

    def _prepare_data(self, G):
        """Converts NetworkX graph to PyTorch Geometric format."""
        # Create node mapping
        incident_nodes = [n for n in G.nodes() if 'labeling' in G.nodes[n]]
        self.node_id_to_idx = {G.nodes[n]['node_id']: i for i, n in enumerate(incident_nodes)}

        # Extract features and labels
        features = []
        labels = []
        train_mask = []
        test_mask = []

        for node in incident_nodes:
            features.append(G.nodes[node]['features'])
            labels.append(G.nodes[node]['severity_level'])
            train_mask.append(G.nodes[node]['labeling'] == 'train')
            test_mask.append(G.nodes[node]['labeling'] == 'test')

        # Build edge index (only between incident nodes)
        edge_index = []
        incident_set = set(incident_nodes)
        for u, v in G.edges():
            if u in incident_set and v in incident_set:
                u_idx = incident_nodes.index(u)
                v_idx = incident_nodes.index(v)
                edge_index.append([u_idx, v_idx])
                edge_index.append([v_idx, u_idx])  # undirected

        # Convert to tensors
        x = torch.tensor(np.array(features), dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    def train(self, G, epochs=100, lr=0.01, early_stopping=10):
        """
        Trains the GraphSAGE model.

        Args:
            G (nx.Graph): The graph to train on.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            early_stopping (int): Patience for early stopping.
        """
        self.data = self._prepare_data(G).to(self.device)

        # Initialize model
        self.model = GraphSAGE(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            out_channels=len(torch.unique(self.data.y))
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        train_counter = self.data.train_mask.sum().item()
        print(f"Num of train nodes: {train_counter}")
        print('\nTraining')

        best_val_acc = 0
        patience_counter = 0

        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            optimizer.zero_grad()

            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                train_acc = (pred[self.data.train_mask] == self.data.y[self.data.train_mask]).float().mean()

                # Use a small portion of training data as validation
                val_size = max(1, train_counter // 20)
                val_indices = torch.where(self.data.train_mask)[0][:val_size]
                val_acc = (pred[val_indices] == self.data.y[val_indices]).float().mean()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

    def predict(self, G):
        """
        Makes predictions on the graph.

        Args:
            G (nx.Graph): The graph to predict on.

        Returns:
            dict: A dictionary containing prediction information.
        """
        if self.model is None:
            raise Exception("Model is not trained yet. Call train() first.")

        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1).cpu().numpy()
            proba = F.softmax(out, dim=1).cpu().numpy()

        # Collect results
        incident_nodes = [n for n in G.nodes() if 'labeling' in G.nodes[n]]
        ids = [G.nodes[n]['node_id'] for n in incident_nodes]
        y_true = [str(G.nodes[n]['severity_level']) for n in incident_nodes]
        labeling = [G.nodes[n]['labeling'] for n in incident_nodes]

        y_pred = [str(p) for p in pred]

        test_mask = self.data.test_mask.cpu().numpy()
        test_ids = [ids[i] for i in range(len(ids)) if test_mask[i]]
        test_y_true = [y_true[i] for i in range(len(y_true)) if test_mask[i]]
        test_y_pred = [y_pred[i] for i in range(len(y_pred)) if test_mask[i]]

        print('Predicting...')
        return {
            "all_ids": ids,
            "all_y_true": y_true,
            "all_y_pred": y_pred,
            "all_y_pred_proba": proba,
            "test_ids": test_ids,
            "test_y_true": test_y_true,
            "test_y_pred": test_y_pred,
            "labeling": labeling
        }
