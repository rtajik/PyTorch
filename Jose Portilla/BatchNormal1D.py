import torch
import torch.nn as nn

bn = nn.BatchNorm1d(3)

print("Gamma:", bn.weight)
print("Beta:", bn.bias)

print("Running Mean:", bn.running_mean)
print("Running Var:", bn.running_var)