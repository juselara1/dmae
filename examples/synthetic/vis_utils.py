"""
Implementation of: Dissimilarity Mixture Autoencoder (DMAE) for Deep Clustering.

**This package contains some visualization utilities that can be used to interpret DMAE.**

Author: Juan Sebastián Lara Ramírez <julara@unal.edu.co> <https://github.com/larajuse>
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def visualize_regions(encoder_model, name, X, figsize=(10, 10), batch_size=32, show_clusters=True):
    # Creating a grid in the data's range
    ran_x = X[:, 0].max()-X[:, 0].min()
    ran_y = X[:, 1].max()-X[:, 1].min()
    x = np.linspace(X[:, 0].min()-ran_x*0.05, X[:,0].max()+ran_x*0.05, 256)
    y = np.linspace(X[:, 1].min()-ran_y*0.05, X[:,1].max()+ran_y*0.05, 256)
    A, B = np.meshgrid(x, y)
    A_flat = A.reshape(-1, 1)
    B_flat = B.reshape(-1, 1)
    X2 = np.hstack([A_flat, B_flat])
    
    # Defining a matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Compute the soft-assignments for the grid
    preds = encoder_model.predict(X2, batch_size=batch_size)
    # Estimate the hard-assignments for the grid
    y_cats2 = np.argmax(preds, axis=1)

    # Compute the soft-assignments for the data
    preds2 = encoder_model.predict(X, batch_size=batch_size)
    # Estimate the hard-assignments for the grid
    y_cats = np.argmax(preds2, axis=1)

    # Show the voronoi regions as an image
    ax.imshow(y_cats2.reshape(A.shape), interpolation='nearest', extent=(x.min(), x.max(), y.min(), y.max()),
               cmap="rainbow", aspect='auto', origin='lower', alpha=0.5)

    ax.axis("off")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(name)

    # Scatter plot of the original points
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4, c="k")
    ax.set_xlim([X2[:, 0].min(), X2[:, 0].max()])
    ax.set_ylim([X2[:, 1].min(), X2[:, 1].max()])
    if show_clusters:
        # Scatter plot of the clusters
        ax.scatter(encoder_model.layers[1].weights[0][:,0], encoder_model.layers[1].weights[0][:,1], c="r")
    return fig, ax

def visualize_distribution(autoencoder, dis_loss, alpha, X, encoder=None, cov=False, figsize=(10, 10)):
    # Creating a grid in covering the data's range
    ran_x = X[:, 0].max()-X[:, 0].min()
    ran_y = X[:, 1].max()-X[:, 1].min()
    x = np.linspace(X[:, 0].min()-ran_x*0.05, X[:,0].max()+ran_x*0.05, 256)
    y = np.linspace(X[:, 1].min()-ran_y*0.05, X[:,1].max()+ran_y*0.05, 256)
    A, B = np.meshgrid(x, y)
    A_flat = A.reshape(-1, 1)
    B_flat = B.reshape(-1, 1)
    X2 = np.hstack([A_flat, B_flat]).astype("float32")
    
    theta_tilde = autoencoder.predict(X2)
    if encoder is None:
        h = X2
    else:
        h = encoder.predict(X2)
    if cov:
        vals = tf.exp(-alpha*dis_loss(h, *theta_tilde)).numpy()
    else:
        vals = tf.exp(-alpha*dis_loss(h, theta_tilde)).numpy()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    vals = vals.reshape(A.shape)
    ax.imshow(vals, interpolation='nearest', extent=(x.min(), x.max(), y.min(), y.max()),
               cmap="Blues", aspect='auto', origin='lower', alpha=0.7)
    ax.axis("off")
    ax.contour(A, B, vals, colors='k', linestyles='-')
    ax.scatter(X[:, 0], X[:, 1], c="k", alpha=0.4)
    return fig, ax

def visualize_probas(encoder, X, n_clusters, figsize=(10, 10), rows=2, cols=2):
    # Creating a grid in covering the data's range
    ran_x = X[:, 0].max()-X[:, 0].min()
    ran_y = X[:, 1].max()-X[:, 1].min()
    x = np.linspace(X[:, 0].min()-ran_x*0.05, X[:,0].max()+ran_x*0.05, 128)
    y = np.linspace(X[:, 1].min()-ran_y*0.05, X[:,1].max()+ran_y*0.05, 128)
    A, B = np.meshgrid(x, y)
    A_flat = A.reshape(-1, 1)
    B_flat = B.reshape(-1, 1)
    X2 = np.hstack([A_flat, B_flat])
    
    
    preds = encoder.predict(X2)
    preds2 = encoder.predict(X)
    y_cats = np.argmax(preds2, axis=1)
    
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(n_clusters):
        vals = preds[:, i].reshape(A.shape)
        axi = ax[i//cols, i%cols] if rows>1 else ax[i]
        axi.imshow(vals, interpolation='nearest', extent=(x.min(), x.max(), y.min(), y.max()),
                   cmap="Blues", aspect='auto', origin='lower', alpha=0.5)
        axi.axis("off")
        axi.set_title(f"$P(z_{i+1}"+"=1|\mathcal{X})$")
        axi.scatter(X[:, 0], X[:, 1], alpha=0.4, c="k")
        axi.contour(A, B, vals, colors='k', linestyles='-')
    return fig, ax