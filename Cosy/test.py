import torch


torch.manual_seed(1986)
torch.cuda.manual_seed_all(1986)

print(torch.rand(1))
print(torch.rand_like(torch.Tensor([1, 3])))


