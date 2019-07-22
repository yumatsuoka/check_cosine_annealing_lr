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

def check_annealing(model, optimizer, param_dict):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=param_dict['t_max'], eta_min=param_dict['eta_min'], last_epoch=-1)

    lr_list = [0. for i in range(param_dict['epochs']) for j in range(param_dict['steps'])]
    for epoch in range(param_dict['epochs']):
        for idx in range(param_dict['steps']):
        
            now_itr = epoch * param_dict['steps'] + idx
            now_lr = scheduler.get_lr()

            lr_list[epoch*steps+idx] = now_lr
            optimizer.step()

            scheduler.step()
            if optimizer.param_groups[0]['lr'] == param_dict['eta_min']:
                if param_dict['whole_decay']:
                    annealed_lr = param_dict['lr'] * (1 + math.cos(
                        math.pi * now_itr / (param_dict['epochs'] * param_dict['steps']) )) / 2
                    optimizer.param_groups[0]['initial_lr'] = annealed_lr
                param_dict['t_max'] *= param_dict['t_mult']
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=param_dict['t_max'], eta_min=param_dict['eta_min'], last_epoch=-1)
                
    return lr_list

epochs = 100
steps = 200
lr = 1.

t01_tmult2 = {
    'epochs':       epochs,
    'steps':        steps,
    't_max':        steps*1,
    't_mult':       2,
    'eta_min':      0,
    'lr':           lr,
    'whole_decay':  False,
    'out_name':     "T_0={}-T_mult={}".format(steps*1, 2),
    }

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Run
t01_tmult2_out = check_annealing(model, optimizer, t01_tmult2)

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
![ZZ](cos_anni.png?raw=true "X")
