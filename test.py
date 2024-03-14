import torch

f0 = torch.load("feature_data/f0_data/serifu/1.pt")

for i in range(f0.size(-1)):
    print(f0[0][i])