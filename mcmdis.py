import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class MLP(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=[256, 256, 256], out_dim=1, act='sigmoid'):
        super(MLP, self).__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_dim:
            layers.append(nn.Linear(last_dim, h))
            layers.append(Act(act))
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='sigmoid')
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
class Act(nn.Module):
    def __init__(self, act):
        super(Act, self).__init__()
        self.act = act
    def forward(self, x):
        if self.act == 'sigmoid':
            return torch.sigmoid(x)
        elif self.act == 'swish':
            return x * torch.sigmoid(x)
        elif self.act == 'elu':
            return F.elu(x)
        elif self.act == 'gelu':
            return F.gelu(x)
        elif self.act == 'tanh':
            return torch.tanh(x)
        elif self.act == 'softplus':
            return F.softplus(x)

for N in [150, 300, 600, 1200, 2500, 5000, 10000]:
    for noise in ([0, 0.05, 0.10] if N == 10000 else [0]):
        for fey in [2, 3, 4]:
            rec = {}
            metadata = pkl.load(open(f'./feynman_interactions_{fey}atoms/metadata.pkl', 'rb'))
            for act in ['softplus', 'sigmoid', 'swish', 'elu', 'gelu', 'tanh']:
                rec[act] = []
                batch_size=N
                aurocs = []
                for bench in tqdm(range(100)):
                    data = pd.read_csv(f'./feynman_interactions_{fey}atoms/test_{bench}.csv')[:N]
                    X = data[[f'x{i}' for i in range(10)]].values
                    y = data['y'].values
                    noise_X = np.random.normal(1, noise, X.shape)
                    noise_y = np.random.normal(1, noise, y.shape)
                    X = torch.tensor(X * noise_X, dtype=torch.float32).to(device)
                    y = torch.tensor(y * noise_y, dtype=torch.float32).to(device)
                    
                    loss_fn = nn.MSELoss()
                    mlp = MLP(act=act).to(device)
                    opt = optim.Adam(mlp.parameters(), lr=1e-3)
                    best = None
                    best_loss = float('inf')
                    num_batches = (N + batch_size - 1) // batch_size
                    for ep in range(10000):
                        perm = torch.randperm(N)
                        X_shuffled = X[perm]
                        y_shuffled = y[perm]
                        for b in range(num_batches):
                            start = b * batch_size
                            end = min(start + batch_size, N)
                            X_batch = X_shuffled[start:end]
                            y_batch = y_shuffled[start:end]
                            y_pred = mlp(X_batch)
                            loss = loss_fn(y_pred.view(-1), y_batch)
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            best = mlp.state_dict()
                    mlp.load_state_dict(best)
                    mlp.eval()
                    M = 100
                    feats = 10
                    avg = np.zeros((feats, feats))
                    der = np.zeros(feats)
                    
                    for i in range(M):
                        x0 = X[i].unsqueeze(0)
                        x0.requires_grad = True
                        y0 = mlp(x0)
                        grad_output = torch.ones_like(y0)
                        grad_input = grad(y0, x0, grad_outputs=grad_output, create_graph=True)[0]
                        hessian = []
                        for g in grad_input[0]:
                            second_grad = grad(g, x0, create_graph=True)[0]
                            hessian.append(second_grad[0].detach().cpu().numpy())
                        hessian = np.abs(np.stack(hessian))
                        avg += hessian/M 
                    for i in range(feats):
                        for j in range(feats):
                            if i <= j: avg[i, j] = 0.
                    
                    gt = []
                    atoms = metadata[bench]['atoms']
                    for a in atoms:
                        if isinstance(a, int): break
                        for i in range(len(a)):
                            for j in range(i + 1, len(a)):
                                if a[i] > a[j]: gt.append((a[i], a[j]))
                                else: gt.append((a[j], a[i]))
                    
                    labels = []
                    scores = []
                    for i in range(feats):
                        for j in range(i):
                            labels.append((i, j) in gt)
                            scores.append(avg[i, j])
                    auroc = roc_auc_score(labels, scores)
                    aurocs.append(auroc)
                print(fey, seed, N, noise, act, sum(aurocs)/100)
