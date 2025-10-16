import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm

TRAIN_DATA_PATH = "gnn_train_data.pt"
MODEL_PATH = "gnn_model.pt"

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def train_model(X, y, hidden_dim=64, epochs=50, lr=0.01):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    data = Data(x=X, edge_index=None, y=y)
    model = GCN(in_dim=X.size(1), hidden_dim=hidden_dim).to(device)
    X, y = X.to(device), y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc="训练GNN"):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index if data.edge_index is not None else torch.arange(X.size(0)).unsqueeze(0).repeat(2,1).to(device))
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ GNN模型训练完成，已保存至 {MODEL_PATH}")

if __name__ == "__main__":
    X, y = torch.load(TRAIN_DATA_PATH)
    train_model(X, y)