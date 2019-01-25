# Check cosine annealing lr on Pytorch
  
I checked the PyTorch implementation of the learning rate scheduler with some learning rate decay conditions.  
torch.optim.lr_scheduler.CosineAnnealingLR()  
https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR  
Which is the implementation of this paper.  
SGDR: Stochastic Gradient Descent with Warm Restarts  
https://arxiv.org/abs/1608.03983  

## Environment
  
Use [Colaboratory](https://colab.research.google.com)  

## Example

```python
import torch

def check_annealing(epochs, steps, model, optimizer, dict):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dict['t_max'], eta_min=0, last_epoch=-1)

    lr_list = [0. for i in range(epochs) for j in range(steps)]
    for epoch in range(epochs):
        for idx in range(steps):

            now_lr = scheduler.get_lr()
            #print(now_lr)
            lr_list[epoch*steps+idx] = now_lr
            optimizer.step()

            scheduler.step()
            if dict['t_max'] * dict['t_mult'] - steps == epoch * steps + idx:
                dict['t_max'] *= dict['t_mult']
                #print('Reset scheduler')
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dict['t_max'], eta_min=0, last_epoch=-1)
    return lr_list

epochs = 100
steps = 200
lr = 1.

t01_tmult2 = {'epochs': epochs,
              'steps': steps,
               't_max': steps*1,
               't_mult': 2,
               'lr': lr,
               'out_name': "T_0={}-T_mult={}".format(steps*1, 2),
              }

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Run
t01_tmult2_out = check_annealing(epochs, steps, model, optimizer, t01_tmult2)


# Visualize
def show_graph(lr_lists, epochs, steps, out_name):
    plt.clf()
    plt.rcParams['figure.figsize'] = [20, 5]
    x = list(range(epochs * steps))
    plt.plot(x, lr_lists, label="line L")
    plt.plot()

    plt.ylim(10e-5, 1)
    plt.yscale("log")
    plt.xlabel("iterations")
    plt.ylabel("learning rate")
    plt.title("Check Cosine Annealing Learing Rate with {}".format(out_name))
    plt.legend()
    plt.show()

show_graph(t01_tmult2_out, epochs, steps, t01_tmult2['out_name'])
```

## Result
![ZZ](t_0-200-t_multi-2.png?raw=true "X")
