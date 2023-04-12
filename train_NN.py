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
    param_list = torch.cat(param_list).detach().cpu().numpy()
    pickle.dump(param_list, open(f"{save_data_location}/Theta_{itr}.p", 'wb'))
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
                   save_data_frequency: int = 100,
                   save_eigs_only: bool = True):
    """ The function that trains a model and saves the Jacobians.
    
    Parameters
        model
            A pytorch neural network model
        train_data
            A tuple (Boundary input, Bounday ground truth, Residual input,
                    residual ground truth)
                    each is a numpy array
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
        save_eigs_only
            Saves the eigenvalues of K_uu, K_rr and K only (to save storage)
    
    Returns
        optimization_details
            A dictionary with the evoluation of the loss components
    """
    # Save model at initialization - for data analysis
    save_model(model, 0, save_data_location)

    # Return to cuda if needed
    model.to(device)

    # Convert data to Pytorch tensors requiring gradient
    X_u, Y_u, X_r, Y_r = train_data
    X_u = torch.tensor(X_u, requires_grad=True).to(device)
    X_r = torch.tensor(X_r, requires_grad=True).to(device)
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

    optimization_details = {'Loss': [],
                            'L_b': [], 'L_r': [],
                            }
    for itr in range(no_iterations):
        # feed data to network
        Y_u_pred = model(X_u)  # (no_of_points, 1)

        if itr % save_data_frequency == 0:
            J_u = utilities.calculate_j_u(model, Y_u_pred)

        Y_r_pred = model(X_r)
        residual = calculate_residual(Y_r_pred, X_r)

        if itr % save_data_frequency == 0:
            J_r = utilities.calculate_j_r(model, residual, device)

        L_r = torch.mean((residual - Y_r_tor) ** 2)
        L_b = torch.mean((Y_u_pred - Y_u_tor) ** 2)
        loss = L_b + L_r

        # optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # log all details  
        optimization_details['Loss'].append(loss.detach().cpu().numpy().item())
        optimization_details['L_r'].append(L_r.detach().cpu().numpy().item())
        optimization_details['L_b'].append(L_b.detach().cpu().numpy().item())

        # Pickle the Jacobians' eigenvalues [saves storage]
        if itr % save_data_frequency == 0:
            # Save eigenvalues only
            if save_eigs_only:
                e_uu, e_rr, e_k = utilities.calculate_eigenvalues(J_u, J_r)            
                np.savez_compressed("{}/eigenvalues_itr_{}".format(save_data_location, itr),
                                    eig_uu=e_uu, eig_rr=e_rr, eig_K=e_k)
            else:
                np.savez_compressed("{}/J_matrices_itr_{}".format(save_data_location, itr),
                                    J_u=J_u, J_r=J_r)
            save_model(model, itr, save_data_location)
    return optimization_details
