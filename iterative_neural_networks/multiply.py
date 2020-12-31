import torch
from typing import List
from collections import defaultdict
import numpy as np
from torch import nn
from torch.nn import functional
from dataclasses import dataclass
import matplotlib
matplotlib.use('MacOSX')
from matplotlib import pyplot as plt

@dataclass
class Itermediate: # puns are a sign of good code, amirite
  inp: torch.Tensor
  hidden: List[torch.Tensor]
  out: torch.Tensor

class IternetMult(nn.Module):
  def __init__(self, width):
    super(IternetMult, self).__init__()
    # constants
    self.width = width
    hidden_inp_size = 2 + 1 + width # 2 true inputs, 1 for 1's, width for hiddens
    self.hidden_w_mask = torch.ones(hidden_inp_size, width)

    # parameters
    self.hidden_w = nn.Parameter(torch.normal(
      mean=torch.zeros(hidden_inp_size, width),
      std=1
    ) * np.sqrt(2.0 / width))
    self.last = nn.Parameter(torch.normal(
      mean=torch.zeros(width, 1),
      std=1.0
    ) * np.sqrt(2.0))


  def init_itermediate(self, inp):
    ones = torch.ones(inp.shape[0])[:, None]
    expanded_inp = torch.cat([ones, inp], dim=1)
    batch_size = inp.shape[0]
    hidden = torch.zeros(batch_size, self.width)
    out = torch.zeros(batch_size, 1)
    return Itermediate(
      expanded_inp,
      hidden,
      out
    )
    

  def forward(self, inp, iters):
    itermediate = self.init_itermediate(inp)
    for _ in range(iters):
      itermediate = self.iterate(itermediate)
    return itermediate.out[:, 0]

  def iterate(self, itermediate):
    inp = torch.cat([itermediate.inp, itermediate.hidden], dim=1)
    adj_hidden_w = self.hidden_w * self.hidden_w_mask
    normalized_hidden_w = adj_hidden_w / torch.sqrt(torch.sum(adj_hidden_w ** 2, dim=0)[None, :])
    hidden = functional.relu(torch.matmul(inp, normalized_hidden_w))
    out = torch.matmul(hidden, self.last)
    return Itermediate(itermediate.inp, hidden, out)
#class IternetMult(nn.Module):
#  def __init__(self, widths):
#    super(IternetMult, self).__init__()
#    self.widths = widths
#    self.layers = []
#    prev_width = 3
#    for i in range(len(widths)):
#      width = widths[i]
#      next_width = widths[i + 1] if i < len(widths) - 1 else 0
#      layer = nn.Parameter(torch.normal(
#        mean=torch.zeros(prev_width + next_width, width),
#        std=1,
#      ) * np.sqrt(2.0 / width))
#      setattr(self, f'l{i}', layer)
#      self.layers.append(layer)
#      prev_width = width
#    self.last = nn.Parameter(torch.normal(
#      mean=torch.zeros(widths[-1], 1),
#      std=1.0
#    ) * np.sqrt(2.0))
#
#  def init_itermediate(self, inp):
#    ones = torch.ones(inp.shape[0])[:, None]
#    expanded_inp = torch.cat([ones, inp], dim=1)
#    batch_size = inp.shape[0]
#    hidden = [torch.zeros(batch_size, width) for width in self.widths]
#    out = torch.zeros(batch_size, 1)
#    return Itermediate(
#      expanded_inp,
#      hidden,
#      out
#    )
#    
#
#  def forward(self, inp, iters):
#    itermediate = self.init_itermediate(inp)
#    for _ in range(iters):
#      itermediate = self.iterate(itermediate)
#    return itermediate.out[:, 0]
#    
#  def iterate(self, itermediate):
#    hidden = []
#    def compute_layer(inp, weight):
#      normalized_weight = weight / torch.sqrt(torch.sum(weight ** 2, dim=0)[None, :])
#      res = functional.relu(torch.matmul(inp, normalized_weight))
#      hidden.append(res)
#
#    current_inp = torch.cat([itermediate.inp, itermediate.hidden[1]], dim=1)
#    compute_layer(current_inp, self.l0)
#
#    for i in range(1, len(self.widths) - 1):
#      current_inp = torch.cat([hidden[i - 1], itermediate.hidden[i + 1]], dim=1)
#      compute_layer(current_inp, self.layers[i])
#
#    compute_layer(hidden[-1], self.layers[-1])
#
#    out = torch.matmul(hidden[-1], self.last)
#    return Itermediate(itermediate.inp, hidden, out)


class DirectMult(nn.Module):
  def __init__(self, widths):
    super(DirectMult, self).__init__()
    self.widths = widths
    
    self.layers = []
    for i in range(len(widths)):
      width = widths[i]
      self.layers.append(
        nn.Parameter(torch.normal(
          mean=torch.zeros(3 if i == 0 else widths[i - 1], widths[i]),
          std=1,
        ) * np.sqrt(2.0 / width))
      )
    self.last = nn.Parameter(torch.normal(
      mean=torch.zeros(width, 1),
      std=1.0
    ) * np.sqrt(2.0))

  def forward(self, inp):
    ones = torch.ones(inp.shape[0])[:, None]
    x = torch.cat([ones, inp], dim=1)
    for i in range(len(self.widths)):
      x = functional.relu(torch.matmul(x, self.layers[i]))
    
    out = torch.matmul(x, self.last)
    return out[:, 0]

model = IternetMult(width=25)
direct = DirectMult(widths=[175])
batch_size = 111
train_iters = 5
loss_fn = nn.MSELoss()
lr = 0.1
train_batches = 1000
max_iters = 20
print('model params:')
for name, p in model.named_parameters():
  print(name)
print('')
losses = defaultdict(list)
for i in range(train_batches):
  model.zero_grad()
  direct.zero_grad()
  xy = torch.normal(
    mean=torch.zeros(batch_size, 2),
    std=1
  )
  z = (xy[:, 0] * xy[:, 1])
  predicted = model(xy, iters=train_iters)
  direct_predicted = direct(xy)
  loss = loss_fn(predicted, z)
  direct_loss = loss_fn(direct_predicted, z)
  loss.backward()
  direct_loss.backward()
  for p in model.parameters():
    p.data.add_(other=p.grad.data, alpha=-lr)
  for p in direct.parameters():
    p.data.add_(other=p.grad.data, alpha=-lr)
  with torch.no_grad():
    eq_loss = loss_fn(model(xy, iters=max_iters), z)
  losses[f'iter {train_iters}'].append(loss.item())
  losses[f'iter {max_iters}'].append(eq_loss.item())
  losses[f'traditional'].append(direct_loss.item())
  s = ''
  for k, v in losses.items():
    s += f'  {k} {v[-1]})'
  print(s)

#print(model.l0.data)
#print(model.l1.data)
for k, seq in losses.items():
  plt.plot(seq, label=k)
plt.legend()
plt.show()

colors = [
  '#aa0000',
  '#dd9900',
  '#bbaa00',
  '#99cc00',
  '#00aa00',
  '#009966',
  '#0066aa',
  '#0000bb',
  '#5500aa',
  '#880066',
]

n = len(colors)
while True:
  xy = torch.normal(
    mean=torch.zeros(n, 2),
    std=1
  )
  z = (xy[:, 0] * xy[:, 1])
  perm = np.argsort(z.data)
  xy = xy[perm]
  z = z[perm]
  itermediate = model.init_itermediate(xy)
  outs = [itermediate.out.data]
  with torch.no_grad():
    for i in range(max_iters):
      itermediate = model.iterate(itermediate)
      outs.append(itermediate.out.data)
  for j in range(n):
    c = colors[j]
    gt = z[j].item()
    plt.axhline(gt, color=c, linestyle='--')
    plt.plot([outs[i][j] for i in range(max_iters)], color=c)
  plt.xlabel('iteration')
  plt.ylabel('prediction')
  plt.axvline(train_iters, label='train iters', color='black', linestyle='--')
  plt.legend()
  plt.show()
