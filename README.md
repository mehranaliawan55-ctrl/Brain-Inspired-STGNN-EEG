# Brain-Inspired Spatio-Temporal Graph Neural Network (ST-GNN) for EEG Decoding

A high-performance PyTorch framework designed for **Brain-Computer Interfaces (BCI)** and cognitive state decoding, integrating **Graph Neural Networks (GNNs)** with **Continual Learning Memory Bottlenecks**.

## 🧠 Research Motivation
Decoding high-dimensional neural signals (EEG/fMRI) requires capturing both complex temporal dynamics and non-Euclidean spatial functional connectivity across brain regions. This repository provides a robust architecture bridging spatio-temporal modeling with brain-inspired representation learning.

## 🌟 Key Architecture Highlights
1. **Temporal Feature Extraction:** Multi-scale 1D convolutions isolate neurophysiological wave rhythms.
2. **Spatial Graph Convolutions:** Models functional dependencies between brain electrodes using graph adjacency matrices.
3. **Continual Learning Bottleneck:** Incorporates structured hidden projections to maintain plasticity and prevent catastrophic forgetting across incremental BCI tasks.

## 📦 Repository Structure
- `model.py`: Core PyTorch definitions for `BrainInspiredSTGNN` and graph-convolution modules.
- `requirements.txt`: Environment dependencies.

## 💻 Code Usage Example
```python
import torch
from model import BrainInspiredSTGNN

# Initialize model for 64-channel EEG data over 500 time points
model = BrainInspiredSTGNN(num_channels=64, time_steps=500, num_classes=4)

# Generate synthetic batch and brain adjacency graph
eeg_batch = torch.randn(8, 64, 500)
brain_adj = torch.eye(64) 

# Forward pass execution
outputs = model(eeg_batch, brain_adj)
print("Predicted Class Logits:", outputs.shape)
```

## 📜 Requirements
- Python >= 3.8
- PyTorch >= 1.12.0
- NumPy >= 1.21.0

## 🤝 Academic Alignment
Targeted toward advanced research in brain-inspired intelligence, neural decoding, and robust multi-modal machine learning frameworks.
