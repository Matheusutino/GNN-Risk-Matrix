# -*- coding: utf-8 -*-
"""
ResultsGenerator module for generating and saving results and risk matrix.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import KBinsDiscretizer


class ResultsGenerator:
    """
    Generates and saves the results and the risk matrix.
    """

    def __init__(self, G, predictions, data_path, results_dir='data/results'):
        self.G = G
        self.predictions = predictions
        self.data_path = data_path
        self.results_dir = results_dir

    def _predict_by_filter(self, filter_key, filter_values):
        results = []

        for value in filter_values:
            ids_test, y_true = [], []
            for node in self.G.nodes():
                if ('labeling' in self.G.nodes[node] and
                    self.G.nodes[node]['labeling'] == 'test' and
                    self.G.nodes[node].get(filter_key) == value):
                    ids_test.append(self.G.nodes[node]['node_id'])
                    y_true.append(str(self.G.nodes[node]['severity_level']))

            if not ids_test: continue

            y_pred = self.predictions["test_y_pred"]
            pred_indices = [self.predictions["test_ids"].index(i) for i in ids_test]
            filtered_y_pred = [y_pred[i] for i in pred_indices]


            acc = accuracy_score(y_true, filtered_y_pred)
            f1 = f1_score(y_true, filtered_y_pred, average='macro', zero_division=0)

            result = {filter_key: value, 'Accuracy': acc, 'F1-Macro': f1}
            results.append(result)

        return pd.DataFrame(results)

    def generate_results(self):
        """
        Generates and returns overall, by-category, and by-risk-type results.
        """
        acc = accuracy_score(self.predictions['test_y_true'], self.predictions['test_y_pred'])
        f1 = f1_score(self.predictions['test_y_true'], self.predictions['test_y_pred'], average='macro')
        general_result = pd.DataFrame({'Accuracy': [acc], 'F1-Macro': [f1]}, index=['Total'])

        df = pd.read_pickle(self.data_path)
        category_result = self._predict_by_filter('product_category', df['Product Category'].unique())
        risk_type_result = self._predict_by_filter('risk_type', df['risk_type'].unique())

        return general_result, category_result, risk_type_result

    def save_results(self, general_result, category_result, risk_type_result):
        """
        Saves the results to CSV and Pickle files.
        """
        general_result.to_csv(os.path.join(self.results_dir, 'csv', 'generalResults.csv'))
        category_result.to_csv(os.path.join(self.results_dir, 'csv', 'categoryResult.csv'), index=False)
        risk_type_result.to_csv(os.path.join(self.results_dir, 'csv', 'riskTypeResult.csv'), index=False)

        general_result.to_pickle(os.path.join(self.results_dir, 'pkl', 'generalResults.pkl'))
        category_result.to_pickle(os.path.join(self.results_dir, 'pkl', 'categoryResult.pkl'))
        risk_type_result.to_pickle(os.path.join(self.results_dir, 'pkl', 'riskTypeResult.pkl'))
        print(f"Results saved to {self.results_dir}")

    def generate_risk_matrix(self):
        """
        Generates the risk matrix dataframe.
        """
        # Centrality calculation
        features = [self.G.nodes[node]['features'] for node in self.G.nodes() if 'labeling' in self.G.nodes[node]]
        knn_graph = kneighbors_graph(features, n_neighbors=5, mode='connectivity', include_self=True)
        new_G = nx.from_numpy_array(knn_graph)
        centrality = nx.closeness_centrality(new_G)

        df = pd.DataFrame()
        incident_nodes = [n for n in self.G.nodes() if 'labeling' in self.G.nodes[n]]

        df['ID'] = [self.G.nodes[n]['node_id'] for n in incident_nodes]
        df['Incident Description'] = [self.G.nodes[n]['incident_description'] for n in incident_nodes]
        df['Product Category'] = [self.G.nodes[n]['product_category'] for n in incident_nodes]
        df['Product Sub Category'] = [self.G.nodes[n]['product_sub_category'] for n in incident_nodes]
        df['Product Type'] = [self.G.nodes[n]['product_type'] for n in incident_nodes]
        df['Risk Type'] = [self.G.nodes[n]['risk_type'] for n in incident_nodes]
        df['Risk Context'] = [self.G.nodes[n]['risk_context'] for n in incident_nodes]
        df['labeling'] = [self.G.nodes[n]['labeling'] for n in incident_nodes]

        # Map predictions back to the dataframe
        pred_map = {id: (pred, proba) for id, pred, proba in zip(self.predictions['all_ids'], self.predictions['all_y_pred'], self.predictions['all_y_pred_proba'])}
        true_map = {id: true for id, true in zip(self.predictions['all_ids'], self.predictions['all_y_true'])}

        df['Predicted Impact'] = df['ID'].map(lambda x: pred_map.get(x, (None, None))[0])
        df['Impact Precision'] = df['ID'].map(lambda x: max(pred_map.get(x, (None, [0]))[1]))
        df['Real Impact'] = df['ID'].map(true_map)

        # Centrality mapping
        centrality_map = {i: c for i, c in centrality.items()}
        df['Closeness Centrality'] = df.index.map(centrality_map)

        # Binning
        est_impact = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
        df['Bin Impact'] = est_impact.fit_transform(df[['Impact Precision']])

        est_centrality = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
        df['Bin Closeness Centrality'] = est_centrality.fit_transform(df[['Closeness Centrality']])

        return df

    def visualize_risk_matrix(self, output_path=None):
        """
        Generates a visual representation of the risk matrix combining severity and probability.

        Args:
            output_path: Path to save the visualization. If None, saves to results_dir/risk_matrix.png
        """
        if output_path is None:
            output_path = os.path.join(self.results_dir, 'risk_matrix.png')

        # Generate the risk matrix dataframe to get the binned data
        df = self.generate_risk_matrix()

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Set the size of the rectangles
        width = 1
        height = 1

        # Matrix order (number of cells in each direction)
        n = 6

        # Define labels for impact and probability categories
        impact = np.array(['Insignificant', 'Minor', 'Significant', 'Major', 'Severe'])
        probability = np.array(['', 'Almost\nCertain', 'Likely', 'Moderate', 'Unlikely', 'Rare'])

        # Calculate count matrix based on binned data
        # Bin Impact: 0-4 (lower = less severe)
        # Bin Closeness Centrality: 0-4 (lower = less probability)
        count_matrix = np.zeros((5, 5))
        for _, row in df.iterrows():
            severity_bin = int(row['Bin Impact'])
            prob_bin = int(row['Bin Closeness Centrality'])
            count_matrix[4 - prob_bin, severity_bin] += 1  # Invert prob for display

        # Add impact labels to the count matrix
        count_matrix = np.vstack((impact, count_matrix))

        # Add probability labels as the initial column
        count_matrix = np.hstack((probability.reshape(-1, 1), count_matrix))

        # Define colors for the rectangles based on risk levels
        color_white = '#F3F7FB'
        color_very_low = '#3AB34A'
        color_low = '#2C8D39'
        color_medium = '#F6E90E'
        color_high = '#F79122'
        color_very_high = '#E91620'
        color_extreme = '#BB121A'

        # Colors
        colors = np.array([
            [color_white, color_white, color_white, color_white, color_white, color_white],
            [color_white, color_medium, color_high, color_very_high, color_extreme, color_extreme],
            [color_white, color_medium, color_medium, color_high, color_very_high, color_extreme],
            [color_white, color_low, color_medium, color_medium, color_high, color_very_high],
            [color_white, color_very_low, color_low, color_medium, color_medium, color_high],
            [color_white, color_very_low, color_very_low, color_low, color_medium, color_medium]
        ])

        # Loop to create the rectangles
        for i in range(n):
            for j in range(n):
                # Calculate the coordinates of the bottom-left corner of the rectangle
                x = j * width
                y = (n - 1 - i) * height

                # Create the rectangle
                rectangle = plt.Rectangle((x, y), width, height, color=colors[i, j], fill=True)

                # Add the rectangle to the plot
                ax.add_patch(rectangle)

                # Display count for data cells, labels for header/first column
                cell_value = count_matrix[i, j]
                if i == 0 or j == 0:  # Header row or column
                    display_text = str(cell_value)
                else:  # Data cells - show count
                    display_text = str(int(float(cell_value)))

                ax.text(x + width / 2, y + height / 2, display_text,
                       ha='center', va='center', fontsize=9)

        # Set the plot limits
        ax.set_xlim(0, n * width)
        ax.set_ylim(0, n * height)

        # Add horizontal grid lines
        for i in range(n + 1):
            ax.axhline(i * height, color='black', linewidth=1)

        # Add vertical grid lines
        for j in range(n + 1):
            ax.axvline(j * width, color='black', linewidth=1)

        # Remove x and y axis labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Show the plot
        plt.grid(True)
        plt.savefig(fname=output_path, dpi=1000, bbox_inches='tight')
        plt.close()

        print(f"Risk matrix visualization saved to {output_path}")
