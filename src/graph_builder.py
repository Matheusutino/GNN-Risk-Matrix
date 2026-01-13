# -*- coding: utf-8 -*-
"""
GraphBuilder module for constructing graphs from processed data.
"""

import pandas as pd
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class GraphBuilder:
    """
    Constructs the graph from the processed data.
    """

    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        """
        Initializes the GraphBuilder.

        Args:
            model_name (str): Name of the sentence transformer model to use.
        """
        self.cache = {}
        self.language_model = SentenceTransformer(model_name)

    def _get_text_features(self, txt):
        if txt in self.cache:
            return self.cache[txt]
        features = self.language_model.encode(txt)
        self.cache[txt] = features
        return features

    def build_graph(self, data_path):
        """
        Builds the NetworkX graph.

        Args:
            data_path (str): Path to the processed data pickle file.

        Returns:
            nx.Graph: The constructed graph.
        """
        df = pd.read_pickle(data_path)
        G = nx.Graph()

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Building graph"):
            incident = f"{index} - {row['snippet']}"
            nodes_to_connect = [
                row['Product Category'],
                row['Product Sub Category'],
                row['Product Type'],
                row['risk_context']
            ]

            for node in [incident] + nodes_to_connect:
                if not G.has_node(node):
                    features = self._get_text_features(row['snippet'] if node == incident else node)
                    G.add_node(node, features=features)

            for node in nodes_to_connect:
                G.add_edge(incident, node)

            G.nodes[incident].update({
                'severity_level': row['severity_level'],
                'labeling': row['labeling'],
                'product_category': row['Product Category'],
                'product_sub_category': row['Product Sub Category'],
                'product_type': row['Product Type'],
                'risk_type': row['risk_type'],
                'risk_context': row['risk_context'],
                'incident_description': row['Incident Description']
            })

        node_id = 1
        for node in G.nodes():
            G.nodes[node]['node_id'] = f'node_id{node_id}'
            node_id += 1

        return G
