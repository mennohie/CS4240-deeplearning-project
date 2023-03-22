import numpy as np
import torch


def calculate_j_u(model, Y_u_pred):
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


def calculate_j_r(model, residual):
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
                g_flat.append(torch.zeros(size=param.size()).view(-1))
        g_flat = torch.cat(g_flat)
        J_r[:, ind] = g_flat 
    return J_r


def calculate_eigenvalues_from_j(j1, j2):
    j1_np = j1.detach().numpy()
    j2_np = j2.detach().numpy()
    K_matrix = j1.T@j2
    eigs = np.linalg.eigvalsh(K_matrix)
    # Return eigenvalues in descenind order
    return eigs[::-1]