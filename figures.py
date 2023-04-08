# import all modules
# Third-party
import numpy as np
import pandas as pd
import torch

# Local files
import utilities
import train_NN as train
import neural_network as net
import pickle
import matplotlib.pyplot as plt


# True PDE - Network will satify this PDE at the end of training
def pde_fn(x, a):
    # PDE: u_xx = -(pi*a)^2.sin(pi*a*x)
    u_xx = -(np.pi * a) ** 2 * np.sin(np.pi * a * x)
    return u_xx


# True solution - Network will try to learn this
def u(x, a):
    return np.sin(np.pi * a * x)


# Create a dataset by sampling points from the PDE's domain
# 1. For the boundaries - subscript u
# 2. Inside the domain = residual - subscript r
# Here, 1D domain, so boundary is just two points!
def run_NTK(a, device, seed=1, max_iterations=100, neurons=500, no_of_data_samples=100):
    bc1_coords = np.array([[0.0],
                           [0.0]])

    print(device)

    bc2_coords = np.array([[1.0],
                           [1.0]])

    dom_coords = np.array([[0.0],
                           [1.0]])

    X_bc1 = dom_coords[0, 0] * np.ones((no_of_data_samples // 2, 1))
    X_bc2 = dom_coords[1, 0] * np.ones((no_of_data_samples // 2, 1))

    X_u = np.vstack([X_bc1, X_bc2])  # data for BC
    Y_u = u(X_u, a)  # y data for BC

    X_r = np.linspace(dom_coords[0, 0],
                      dom_coords[1, 0], no_of_data_samples)[:, None]  # data for residual
    Y_r = pde_fn(X_r, a)  # y data for residual

    # values for normalizing teh datasets
    # Normalize the inputs - X_u and X_r - (n_data_points, dimension)
    mean = X_r.mean(axis=0)  # (1, dim) values
    std = X_r.std(axis=0)  # (1, dim) values

    # Define model and a random seed
    model = net.PINN(no_of_neurons=neurons, no_of_h_layers=2,
                     mean=mean.item(), std=std.item())
    model.to(device)

    # Train the model -> default SGD with full batch and exponential weight decay
    save_data_location = './data_ntk/'
    details = train.train_nn_model(model, train_data=(X_u, Y_u, X_r, Y_r),
                                   no_iterations=max_iterations, device=device,
                                   save_data_location=save_data_location,
                                   save_data_frequency=1)

    # load the jacobian matrices from hard disk
    J_u, J_r = pickle.load(open("{}/Ju_Jr_{}.p".format(save_data_location, 0),
                                'rb'))
    eigs = utilities.calculate_eigenvalues(J_u, J_r)
    return eigs


def get_eigs(neurons=20):
    # Choose a device to run on - either CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ground_truths = [1, 2, 4]
    eigs_all = {}
    eigs_uu = {}
    eigs_rr = {}
    for a in ground_truths:
        print(f"Running network for a={a}")
        eigs = run_NTK(a, device, neurons=neurons, max_iterations=1)
        eigs_all[a] = eigs[0]
        eigs_uu[a] = eigs[1]
        eigs_rr[a] = eigs[2]

    return eigs_all, eigs_uu, eigs_rr
