import numpy as np
import torch
import utilities


# Poisson 1d
a = 4
def u(x, a):
    return np.sin(np.pi * a * x)

def u_xx(x, a):
    return -(np.pi * a)**2 * np.sin(np.pi * a * x)


def calculate_residual(Y, X):
    u_x_val = torch.autograd.grad(
                Y, X, 
                grad_outputs=torch.ones_like(Y),
                retain_graph=True,
                create_graph=True
                )[0] 
    u_xx_val = torch.autograd.grad(
                u_x_val, X,
                grad_outputs=torch.ones_like(u_x_val),
                retain_graph=True,
                create_graph=True
                )[0]
    return u_xx_val
    
    
# Training function
def train_nn_model(model, train_data, no_iterations,
                   optimizer_details={'opt': 'sgd', 'lr': 1e-5}):
    X_u, Y_u, X_r, Y_r = train_data    
    # Normalize the inputs - X_u and X_r - (n_data_points, dim)
    mean = X_r.mean(axis=0)  # (1, dim) values
    std = X_r.std(axis=0)  # (1, dim) values   
    X_u_normal = (X_u - mean) / std
    X_r_normal = (X_r - mean) / std
    # Outputs are not normalized
    # Convert data to Pytorch tensors requiring gradient
    X_u_normal_tor = torch.tensor(X_u_normal, requires_grad=True)
    X_r_normal_tor = torch.tensor(X_r_normal, requires_grad=True)
    Y_u_tor = torch.tensor(Y_u)
    Y_r_tor = torch.tensor(Y_r)
    # TODO: Should we do mini-batching?
    # Start training loop - assuming full-batch
    if optimizer_details['opt'] =='sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                lr=optimizer_details['lr'])
        # Todo: learning rate decay 
    else:
        raise NotImplementedError
    optimization_details = {'Loss':[],
                            'L_b':[], 'L_r':[]
                           }
    # Todo : save jacobian only every 100 steps or so?
    for itr in range(no_iterations):
        # feed data to network
        Y_u_pred = model(X_u_normal_tor)  # (no_of_points, 1)
        J_u = utilities.calculate_j_u(model, Y_u_pred)
        Y_r_pred = model(X_r_normal_tor)
        residual = calculate_residual(Y_r_pred, X_r_normal_tor)
        J_r = utilities.calculate_j_r(model, residual)
        L_r = torch.mean((residual - Y_r_tor)**2)
        L_b = torch.mean((Y_u_pred - Y_u_tor)**2)
        loss = L_b + L_r
        # optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        # log all details  
        optimization_details['Loss'].append(loss.detach().numpy().item())
        optimization_details['L_r'].append(L_b.detach().numpy().item())
        optimization_details['L_b'].append(L_r.detach().numpy().item())        
    return optimization_details