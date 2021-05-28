from utils import *
import torch

num_classes = 3
image_size = [4, 4]

pred = torch.randint(0, num_classes, image_size)
label = torch.randint(0, num_classes, image_size)

print(pred)
print(label)

n_i, n_u, n_p, n_u = intersect_and_union(pred, label, num_classes, 255)
print(n_i, n_u, n_p, n_u)