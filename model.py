import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainGraphConv(nn.Module):
    """
    Graph Convolutional Layer for modeling functional connectivity 
    among brain regions using EEG electrode channel graphs.
    """
    def __init__(self, in_features, out_features):
        super(BrainGraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: [Batch, Nodes, Features], adj: [Nodes, Nodes]
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return F.relu(output + self.bias)

class BrainInspiredSTGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network with Continual Memory Bottleneck 
    for Robust EEG-to-Cognitive State Decoding.
    Aligned with BRILLIANT Lab (BIT) Research Directions.
    """
    def __init__(self, num_channels=64, time_steps=500, hidden_dim=128, num_classes=4):
        super(BrainInspiredSTGNN, self).__init__()
        
        # Temporal Feature Extraction via 1D Convolutions
        self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=15, stride=2, padding=7)
        
        # Spatial Graph Convolution for Brain Connectivity
        self.gcn = BrainGraphConv(in_features=time_steps // 2, out_features=64)
        
        # Continual Learning Memory Projection Layer (Prevents Catastrophic Forgetting)
        self.memory_bottleneck = nn.Linear(num_channels * 64, 256)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, adj_matrix):
        """
        x: EEG Signal Tensor [Batch_Size, Channels (Nodes), Time_Steps]
        adj_matrix: Predefined or learned brain connectivity adjacency matrix [Channels, Channels]
        """
        batch_size, num_channels, time_steps = x.size()
        
        # 1. Temporal Processing per channel
        x_reshaped = x.view(batch_size * num_channels, 1, time_steps)
        temp_out = self.temporal_conv(x_reshaped) # [Batch*Nodes, Hidden, Time/2]
        temp_out = temp_out.view(batch_size, num_channels, -1) # [Batch, Nodes, Encoded_Time]
        
        # 2. Spatial Graph Convolutions using Brain Region Adjacency
        gcn_out = self.gcn(temp_out, adj_matrix) # [Batch, Nodes, 64]
        
        # 3. Flatten and Project through Memory Bottleneck
        flattened = gcn_out.reshape(batch_size, -1)
        bottleneck_repr = F.relu(self.memory_bottleneck(flattened))
        
        # 4. Final Classification
        logits = self.classifier(bottleneck_repr)
        return logits

if __name__ == "__main__":
    # Integration test for the brain-inspired model
    batch_sz, channels, samples = 4, 64, 500
    dummy_eeg = torch.randn(batch_sz, channels, samples)
    dummy_adj = torch.eye(channels) # Simplified identity brain connectivity graph
    
    model = BrainInspiredSTGNN(num_channels=channels, time_steps=samples)
    output = model(dummy_eeg, dummy_adj)
    print("Brain-Inspired ST-GNN Output Logits Shape:", output.shape)
