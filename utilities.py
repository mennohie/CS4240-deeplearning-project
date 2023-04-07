import numpy as np
import torch


def calculate_j_u(model, Y_u_pred):
    """Calculates the jacobian for the boundary condition w.r.t.
        model parameters.
    model
        Pytorch neural network model
    Y_u_pred
        The predictions for the boundary condition points
    """
    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    n_u_batchsize = Y_u_pred.shape[0]
    J_u = torch.zeros(size=(n_params, n_u_batchsize))
    for ind in range(n_u_batchsize):
        grad = torch.autograd.grad(
            Y_u_pred[ind, 0], model.parameters(),
            grad_outputs=torch.ones_like(Y_u_pred[ind, 0]),
            retain_graph=True,
            create_graph=False
        )
        g_flat = []
        for g_component in grad:
            g_flat.append(g_component.view(-1))
        g_flat = torch.cat(g_flat)
        J_u[:, ind] = g_flat
    return J_u


def calculate_j_r(model, residual, device):
    """Calculates the jacobian of the residual w.r.t. model parameters.
    model
        Pytorch neural network model
    residual
        The prediction of the network for the residual.
    """
    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    n_r_batchsize = residual.shape[0]
    J_r = torch.zeros(size=(n_params, n_r_batchsize))
    for ind in range(n_r_batchsize):
        model.zero_grad()
        residual[ind, 0].backward(retain_graph=True)
        g_flat = []
        for param in model.parameters():
            if param.grad is not None:
                g_flat.append(param.grad.view(-1))
            else:
                g_flat.append(torch.zeros(size=param.size(),
                                          device=device).view(-1))
        g_flat = torch.cat(g_flat)
        J_r[:, ind] = g_flat
    return J_r


def calculate_eigenvalues(ju, jr):
    """ Calculates the eigenvalues of the kernel matrix.
    K = j1.T@j2,
    WARNING : 
        Make sure that ju and jr have the shape (n_model_parameters, batch_size)
    
    Returns
        The eigenvalues in descending order for K_uu, K_rr and K.
        See paper Fig 1.
    """
    k_uu_eigvals = np.linalg.eigvalsh(ju.T@ju)
    k_rr_eigvals = np.linalg.eigvalsh(jr.T@jr)
    # The main J matrix ->  The shape is (2* batch_size, no_of_parameters) 
    J = np.vstack((ju.T, jr.T))
    # Operations are correct -> This was done to maintain correct dimensionality
    K_eigvals = np.linalg.eigvalsh(J@J.T)
    return k_uu_eigvals[::-1], k_rr_eigvals[::-1], K_eigvals[::-1]
