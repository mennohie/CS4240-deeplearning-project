import numpy as np
import torch
import utilities
import pickle

def save_model(model: torch.nn.Module, itr: int, 
               save_data_location: str):
    """ Saves model parameters as list to the hard disk.
    model
        A pytorch neural network model
    itr
        The iteration- Used to name the file only
    save_data_location
        The path to the folder to save the parameters    
    """
    param_list = []
    for param in model.parameters():
        param_list.append(param.view(-1))
    param_list = torch.cat(param_list).detach().numpy()
    pickle.dump(param_list, open("{}/Theta_{}.p".format(save_data_location,
                                           itr), 'wb'))
    return None


def calculate_residual(Y, X):
    """Calculates u_xx using an existing graph.
    Y
        Output predicted by network
    X 
        Input to the network
    """
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
def train_nn_model(model: torch.nn.Module, train_data: tuple, 
                   no_iterations: int, device, 
                   optimizer_details: dict = {'opt': 'sgd', 'lr': 1e-5},
                   save_data_location: str = '.',
                   save_data_frequency: int = 100):
    """ The function that trains a model and saves the Jacobians.
    
    Parameters
        model
            A pytorch neural network model
        train_data
            A tuple (Boundary input, Bounday ground truth, Residual input,
                    residual ground truth)
        no_iterations
            Maximum number of epochs to train for [Assuming full batch]
        device
            Whether to run on GPU or CPU - Takes a pytorch device object
        optimizer_details
            A dictionary with the details of the optimizer
            Currently, only SGD is implemented with a exponential 
            learning rate decay
        save_data_location
            Path to folder to save the model parameters and jacobians
        save_data_frequency
            The frequency with whcih details are saved onto the hard disk
    
    Returns
        optimization_details
            A dictionary with the evoluation of the loss components
    """
    # Save model at initialization - for data analysis
    save_model(model, 0, save_data_location)
    
    X_u, Y_u, X_r, Y_r = train_data   
    # Normalize the inputs - X_u and X_r - (n_data_points, dimension)
    mean = X_r.mean(axis=0)  # (1, dim) values
    std = X_r.std(axis=0)  # (1, dim) values   
    X_u_normal = (X_u - mean) / std
    X_r_normal = (X_r - mean) / std
    # Outputs are not normalized
    # Convert data to Pytorch tensors requiring gradient
    X_u_normal_tor = torch.tensor(X_u_normal, 
                                  requires_grad=True).to(device)
    X_r_normal_tor = torch.tensor(X_r_normal, 
                                  requires_grad=True).to(device)
    Y_u_tor = torch.tensor(Y_u).to(device)
    Y_r_tor = torch.tensor(Y_r).to(device)
    # TODO: Should we do mini-batching - Not yet!
    
    # Start training loop - assuming full-batch
    if optimizer_details['opt'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=optimizer_details['lr'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=0.9)
    else:
        raise NotImplementedError
    
    optimization_details = {'Loss':[],
                            'L_b':[], 'L_r':[], 
                           }
    for itr in range(no_iterations):
        # feed data to network
        Y_u_pred = model(X_u_normal_tor)  # (no_of_points, 1)
        
        if itr % save_data_frequency == 0:
            J_u = utilities.calculate_j_u(model, Y_u_pred)
            
        Y_r_pred = model(X_r_normal_tor)
        residual = calculate_residual(Y_r_pred, X_r_normal_tor)
        
        if itr % save_data_frequency == 0:
            J_r = utilities.calculate_j_r(model, residual)
            
        L_r = torch.mean((residual - Y_r_tor)**2)
        L_b = torch.mean((Y_u_pred - Y_u_tor)**2)
        loss = L_b + L_r
        
        # optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        scheduler.step()
        
        # log all details  
        optimization_details['Loss'].append(loss.detach().numpy().item())
        optimization_details['L_r'].append(L_b.detach().numpy().item())
        optimization_details['L_b'].append(L_r.detach().numpy().item())
        
        # Pickle the Jacobians
        if itr % save_data_frequency == 0:
            pickle.dump([J_u, J_r], open("{}/Ju_Jr_{}.p".format(
                                        save_data_location, itr), 'wb'))  
            save_model(model, itr, save_data_location)   
    return optimization_details