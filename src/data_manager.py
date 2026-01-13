# -*- coding: utf-8 -*-
"""
DataManager module for handling downloading, preprocessing, and saving data.
"""

import os
import pandas as pd
import nltk
from tqdm import tqdm


class DataManager:
    """
    Handles downloading, preprocessing, and saving the data.
    """

    def __init__(self, data_dir='data'):
        """
        Initializes the DataManager.

        Args:
            data_dir (str): The directory to store data and results.
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.datasets_dir = os.path.join(data_dir, 'datasets')
        self.results_dir = os.path.join(data_dir, 'results')
        self.csv_results_dir = os.path.join(self.results_dir, 'csv')
        self.pkl_results_dir = os.path.join(self.results_dir, 'pkl')

    def setup_directories(self):
        """
        Creates all necessary directories.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.csv_results_dir, exist_ok=True)
        os.makedirs(self.pkl_results_dir, exist_ok=True)

    def check_raw_data(self):
        """
        Checks if the raw data files exist in data/raw directory.
        Raises FileNotFoundError if any required file is missing.
        """
        injuries_file = os.path.join(self.raw_dir, 'injuries_expansion_taxonomy.csv')
        incidents_file = os.path.join(self.raw_dir, 'IncidentReports.csv')

        missing_files = []
        if not os.path.exists(injuries_file):
            missing_files.append(injuries_file)
        if not os.path.exists(incidents_file):
            missing_files.append(incidents_file)

        if missing_files:
            raise FileNotFoundError(
                f"Missing required data files: {missing_files}\n"
                f"Please download the files according to data/README.md instructions."
            )

    def preprocess_data(self, training_size=300):
        """
        Preprocesses the raw data to create a labeled dataset for the GNN.

        Args:
            training_size (int): The number of samples per class for the training set.
        """
        self.check_raw_data()

        injuries_file = os.path.join(self.raw_dir, 'injuries_expansion_taxonomy.csv')
        incidents_file = os.path.join(self.raw_dir, 'IncidentReports.csv')

        df_injuries = pd.read_csv(injuries_file, header=None, sep=';')
        df_injuries.columns = ['injury', 'query']

        df_safety = pd.read_csv(incidents_file, skiprows=[0], encoding='iso8859-2')

        # Extract snippets
        nltk.download('punkt', quiet=True)
        L = []
        for _, row in tqdm(df_safety.iterrows(), total=len(df_safety), desc="Extracting snippets"):
            text = row['Incident Description']
            if pd.notna(text):
                sent_text = nltk.sent_tokenize(text)
                for sentence in sent_text:
                    d = row.to_dict()
                    d['snippet'] = sentence
                    L.append(d)
        df_data2 = pd.DataFrame(L)

        # Select data based on injuries taxonomy
        L = []
        for _, row in tqdm(df_injuries.iterrows(), total=len(df_injuries), desc="Filtering by injury"):
            q = [row['injury'], row['query']]
            df_temp = df_data2[df_data2['snippet'].str.contains(f' {q[1]}', na=False)].drop_duplicates()
            if not df_temp.empty:
                df_temp = df_temp.copy()
                df_temp['risk_type'] = q[0]
                df_temp['risk_context'] = q[1]
                L.append(df_temp)

        df_safety = pd.concat(L)

        # Map severity
        severity_map = {
            'No Incident, No Injury': 0,
            'Unspecified': 0,
            'Incident, No Injury': 0,
            'Injury, No First Aid or Medical Attention Received': 1,
            'Injury, Level of care not known': 1,
            'Injury, Seen by Medical Professional': 1,
            'Injury, First Aid Received by Non-Medical Professional': 1,
            'Injury, Hospital Admission': 1,
            'Injury, Emergency Department Treatment Received': 1,
            'Death': 1
        }
        df_safety['severity_level'] = df_safety['(Primary) Victim Severity'].map(severity_map).fillna(-1).astype(int)

        # Create train/test split
        df_safety = df_safety.sample(frac=1, random_state=42)
        label_counters = {}
        labels = []
        for _, row in df_safety.iterrows():
            label = row['severity_level']
            if label not in label_counters:
                label_counters[label] = 0
            if label_counters.get(label, 0) < training_size:
                labels.append('train')
                label_counters[label] = label_counters.get(label, 0) + 1
            else:
                labels.append('test')
        df_safety['labeling'] = labels

        df_safety.to_pickle(os.path.join(self.datasets_dir, 'dataset_random.pkl'))
        print(f"Processed data saved to {os.path.join(self.datasets_dir, 'dataset_random.pkl')}")
