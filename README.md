# Brain-Inspired Spatio-Temporal Graph Neural Network (STGNN) for EEG Decoding

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Domain](https://img.shields.io/badge/Domain-BCI%20%26%20NeuroAI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A high-performance PyTorch framework designed for **Brain-Computer Interfaces (BCI)** and cognitive state decoding, integrating **Graph Neural Networks (GNNs)** with **Continual Learning Memory Bottlenecks** to process multi-channel EEG signals without catastrophic forgetting.

---

##  Architecture Pipeline

```text
[ Raw EEG Signals ] ➔ (Multi-Scale 1D CNN) ➔ Temporal Feature Maps
                                                    │
                                                    ▼
[ Functional Connectivity ] ➔ (Graph Spatial Conv / GNN) ➔ Spatial Embeddings
                                                    │
                                                    ▼
[ Continual Learning Bottleneck ] ➔ Prevents Catastrophic Forgetting
                                                    │
                                                    ▼
                                      [ Cognitive State Logits ]
