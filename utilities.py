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
                g_flat.append(torch.zeros(size=param.size(), device=device).view(-1))
        g_flat = torch.cat(g_flat)
        J_r[:, ind] = g_flat
    return J_r


def calculate_eigenvalues_from_j(j1, j2):
    """ Calculates the eigenvalues of the kernel matrix.
    K = j1.T@j2,
    make sure that j1 has the shape (n_model_parameters, batch_size)
    returns the eigenvalues in descending order for a (batch_size1, batch_size2)
    matrix
    """
    j1_np = j1.detach().numpy()
    j2_np = j2.detach().numpy()
    K_12 = j1_np @ j2_np.T
    K_matrix = np.concatenate(
        [np.concatenate([j1_np @ j1_np.T, K_12], axis=0),
         np.concatenate([K_12, j2_np @ j2_np.T], axis=0)],
        axis=1)
    eigs = np.linalg.eigvalsh(K_matrix)
    del K_matrix  # Hogs memory
    del K_12  # Hogs memory
    # Return eigenvalues in descending order
    eigs_11 = np.linalg.eigvalsh(j1_np @ j1_np.T)
    eigs_22 = np.linalg.eigvalsh(j2_np @ j2_np.T)

    return eigs[::-1], eigs_11[::-1], eigs_22[::-1]
