# GNN for build a risk matrix

This project aims to construct a risk matrix using a graph neural network (GNN) for classification of documents provided by Safe Products. For this, techniques of sentiment analysis and natural language processing were used.

## Project Structure

```
GNN-Risk-Matrix/
├── src/
│   ├── data_manager.py       # Data downloading and preprocessing
│   ├── graph_builder.py      # Graph construction from processed data
│   ├── gnn_model.py          # GraphSAGE model training and prediction
│   ├── results_generator.py  # Results and risk matrix generation
│   └── orchestrator.py       # Main pipeline orchestrator
├── data/                     # Generated data and results
├── requirements.txt          # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to use

The pipeline is fully automated and can be executed with a single command:

```bash
cd src
python orchestrator.py
```

This will:
1. Download and prepare the datasets
2. Preprocess the data
3. Build the graph
4. Train the GraphSAGE model
5. Generate predictions
6. Save results and risk matrix

## Output

Results are saved in the `data/results/` directory:
- `csv/` - Results in CSV format
- `pkl/` - Results in pickle format, including the risk matrix

## Customization

You can customize the pipeline by modifying parameters in `orchestrator.py`:
- `training_size`: Number of samples per class for training
- `epochs`: Number of training epochs
- `model_name`: Sentence transformer model (default: 'paraphrase-MiniLM-L6-v2')

## Citation

This work is currently under review. Citation information will be provided upon publication.

## Contact

Contact information is omitted for double-blind review.
