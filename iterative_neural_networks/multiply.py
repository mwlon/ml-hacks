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
  hidden: torch.Tensor
  bottleneck: torch.Tensor
  out: torch.Tensor

class IternetMult(nn.Module):
  def __init__(self, width, bottleneck_width, bias=True):
    super(IternetMult, self).__init__()
    # constants
    self.width = width
    self.bottleneck_width = bottleneck_width
    self.bias = bias
    hidden_inp_size = 2 + bottleneck_width # 2 true inputs, width for hiddens
    if bias:
      hidden_inp_size += 1

    # parameters
    self.wih = nn.Parameter(torch.normal(
      mean=torch.zeros(hidden_inp_size, width),
      std=1
    ) * np.sqrt(2.0 / width))
    self.whb = nn.Parameter(torch.normal(
      mean=torch.zeros(width, bottleneck_width),
      std=1
    ) * np.sqrt(2.0 / bottleneck_width))
    self.why = nn.Parameter(torch.normal(
      mean=torch.zeros(width, 1),
      std=1.0
    ) * np.sqrt(2.0))


  def init_itermediate(self, inp):
    if self.bias:
      ones = torch.ones(inp.shape[0])[:, None]
      inp = torch.cat([ones, inp], dim=1)
    batch_size = inp.shape[0]
    hidden = torch.zeros(batch_size, self.width)
    bottleneck = torch.zeros(batch_size, self.bottleneck_width)
    out = torch.zeros(batch_size, 1)
    return Itermediate(
      inp,
      hidden,
      bottleneck,
      out
    )
    

  def forward(self, inp, iters):
    itermediate = self.init_itermediate(inp)
    for _ in range(iters):
      itermediate = self.iterate(itermediate)
    return itermediate.out[:, 0]

  def iterate(self, itermediate):
    hidden_inp = torch.cat([itermediate.inp, itermediate.bottleneck], dim=1)
    hidden = functional.relu(torch.matmul(hidden_inp, self.wih))
    norm_bottle_weight = self.whb / torch.norm(self.whb, dim=0)[None, :]
    bottleneck = torch.matmul(hidden, norm_bottle_weight)
    out = torch.matmul(hidden, self.why)
    return Itermediate(itermediate.inp, hidden, bottleneck, out)


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
      setattr(self, f'l{i}', self.layers[-1])
    self.last = nn.Parameter(torch.normal(
      mean=torch.zeros(width, 1),
      std=1.0
    ) * np.sqrt(2.0))

  def forward(self, inp, **kwargs):
    ones = torch.ones(inp.shape[0])[:, None]
    x = torch.cat([ones, inp], dim=1)
    for i in range(len(self.widths)):
      x = functional.relu(torch.matmul(x, self.layers[i]))
    
    out = torch.matmul(x, self.last)
    return out[:, 0]


def new_main():
  return IternetMult(width=70, bottleneck_width=3, bias=True)
model = new_main()
models = {
  'direct': DirectMult(widths=[175]),
  'direct_1': DirectMult(widths=[175]),
  'direct_2': DirectMult(widths=[175]),
  'main': model,
  'main_1': new_main(),
  'main_2': new_main(),
}
batch_size = 444
train_iters = 5
loss_fn = nn.MSELoss()
lr = 0.03
train_batches = 1000
max_iters = 20
print('model params:')
for name, p in model.named_parameters():
  print(name)
print('')
losses = defaultdict(list)
for i in range(train_batches):
  xy = torch.normal(
    mean=torch.zeros(batch_size, 2),
    std=1
  )
  z = (xy[:, 0] * xy[:, 1])
  cos_mult = 1.00001 + np.cos(i * np.pi / train_batches)
  s = f'{i=}'
  for k, m in models.items():
    m.zero_grad()
    predicted = m(xy, iters=train_iters)
    loss = loss_fn(predicted, z)
    loss.backward()
    for p in m.parameters():
      p.data.add_(other=p.grad.data, alpha=-lr)
    losses[k].append(loss.item())
    s += f'  {k} {loss.item():.6f})'
  print(s)

print('\nEVAL\n=====')
eval_size = 10000
xy = torch.normal(
  mean=torch.zeros(eval_size, 2),
  std=1
)
z = (xy[:, 0] * xy[:, 1])
with torch.no_grad():
  for k, m in models.items():
    predicted = m(xy, iters=train_iters)
    loss = loss_fn(predicted, z)
    full_k = '  ' + k + ' ' * (10 - len(k))
    print(f'  {full_k} {loss.item():.6f})')

#for k, m in models.items():
#  if 'main' in k:
#    print(f'\n{k} weights ====')
#    print(m.wih.data)
#    print(m.whb.data)
#    print(m.why.data)

for k, seq in losses.items():
  plt.plot(seq, label=k)
plt.legend()
plt.show()

smoothing = 10
def smooth(seq):
  return np.mean(np.reshape(seq, [-1, smoothing]), axis=1)
for k, seq in losses.items():
  smoothed_x = smooth(np.arange(len(seq)))
  smoothed_y = smooth(seq)
  plt.plot(smoothed_x, smoothed_y, label=k)
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
