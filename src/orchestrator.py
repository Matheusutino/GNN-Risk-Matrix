#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator module for managing the entire GNN Risk Matrix pipeline.
"""

import os
from data_manager import DataManager
from graph_builder import GraphBuilder
from gnn_model import GNNModel
from results_generator import ResultsGenerator


class Orchestrator:
    """
    Orchestrates the entire GNN Risk Matrix pipeline.
    """

    def run(self):
        """
        Executes the full pipeline.
        """
        print("--- Starting Data Preparation ---")
        dm = DataManager()
        dm.setup_directories()
        dm.preprocess_data()

        print("\n--- Starting Graph Construction ---")
        data_path = os.path.join(dm.datasets_dir, 'dataset_random.pkl')
        gb = GraphBuilder()
        G = gb.build_graph(data_path)

        print("\n--- Starting Model Training ---")
        model = GNNModel()
        model.train(G, epochs=20)

        print("\n--- Generating Predictions ---")
        predictions = model.predict(G)

        print("\n--- Generating and Saving Results ---")
        rg = ResultsGenerator(G, predictions, data_path)
        general, category, risk_type = rg.generate_results()
        rg.save_results(general, category, risk_type)

        print("\n--- General Results ---")
        print(general)

        print("\n--- Generating Risk Matrix ---")
        risk_matrix = rg.generate_risk_matrix()
        risk_matrix_path = os.path.join(dm.results_dir, 'pkl', 'risk_matrix.pkl')
        risk_matrix.to_pickle(risk_matrix_path)
        print(f"Risk matrix saved to {risk_matrix_path}")
        print(risk_matrix.head())


if __name__ == '__main__':
    orchestrator = Orchestrator()
    orchestrator.run()
