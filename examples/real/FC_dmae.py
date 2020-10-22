import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import sys, os
import DMAE

def autoencoder(encoder_dims, decoder_dims, latent_dim, input_layer,
                activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., mode='fan_in',
                                                                                            distribution='uniform')):
    lay = input_layer
    # Fully-connected encoder
    for i, dim in enumerate(encoder_dims):
        lay = tf.keras.layers.Dense(dim, activation=activation, kernel_initializer=kernel_initializer)(lay)
    lay = tf.keras.layers.Dense(latent_dim, kernel_initializer=kernel_initializer)(lay)
    encoder_model = tf.keras.Model(inputs=input_layer, outputs=lay)
    # Fully-connected decoder
    dec_inp = tf.keras.layers.Input(shape=(latent_dim, ))
    lay = dec_inp
    for i, dim in enumerate(decoder_dims[:-1]):
        lay = tf.keras.layers.Dense(dim, activation=activation, kernel_initializer=kernel_initializer)(lay)
    lay = tf.keras.layers.Dense(decoder_dims[-1], activation="linear", kernel_initializer=kernel_initializer)(lay)
    decoder_model = tf.keras.Model(inputs=dec_inp, outputs=lay)
    # Fully-connected autoencoder
    ae_model = tf.keras.Model(inputs=input_layer, outputs=decoder_model(encoder_model(input_layer)))
    return encoder_model, decoder_model, ae_model

def deep_dmae(latent_dim, n_clusters, encoder_model, decoder_model, init_dmae,
              dis, dis_loss, pretrainer, input_shape, cov, X_latent):
    alpha = 1000
    # Defining DMAE Model
    inp = tf.keras.layers.Input(shape=input_shape)
    h = encoder_model(inp)
    
    # DMAE layer
    if cov:
        theta_tilde = DMAE.Layers.DissimilarityMixtureAutoencoderCov(alpha, n_clusters, dissimilarity=dis,
                                                                     trainable={"centers":True, "cov":True, "mixers":True},
                                                                     grad_modifier=1,
                                                                     initializers=init_dmae(pretrainer, X_latent))(h)
    else:
        theta_tilde = DMAE.Layers.DissimilarityMixtureAutoencoder(alpha, n_clusters, dissimilarity=dis,
                                                                  trainable={"centers":True, "mixers":True},
                                                                  initializers=init_dmae(pretrainer))(h)
    x_tilde = decoder_model(theta_tilde[0])
    full_model = tf.keras.Model(inputs=[inp], outputs=x_tilde)
    
    loss1 = dis_loss(h, *theta_tilde, alpha)
    loss2 = tf.losses.mse(inp, x_tilde)
    loss = 0.01*loss1+1.0*loss2
    full_model.add_loss(loss)
    # Defining a deep encoder
    if cov:
        assigns = DMAE.Layers.DissimilarityMixtureEncoderCov(alpha, n_clusters, dissimilarity=dis)(h)
    else:
        assigns = DMAE.Layers.DissimilarityMixtureEncoder(alpha, n_clusters, dissimilarity=dis)(h)
    full_encoder = tf.keras.Model(inputs=[inp], outputs=assigns)
    return full_model, full_encoder

def train(ds_aug, ds_cluster, X, y, da, train_batch, test_batch, trials, make_autoencoder, make_dmae,
          pretrain_optimizer, cluster_optimizer, make_pretrainer, pretrain_params, cluster_params):
    # Lists to store the ae_kmeans results
    accs1 = []; nmis1 = []; aris1 = []
    # Lists to store the dmae results
    accs2 = []; nmis2 = []; aris2 = []
    
    for i in range(trials):
        # Defining autoencoder model
        encoder_model, decoder_model, ae_model = make_autoencoder()
        ae_model.compile(loss="mse", optimizer=pretrain_optimizer["type"](**pretrain_optimizer["params"]))
        # Pretraining the model
        if da:
            ae_model.fit(ds_aug, **pretrain_params)
        else:
            ae_model.fit(X, X, **pretrain_params)
        # Data representation in the latent space
        X_latent = encoder_model.predict(X, batch_size=test_batch)
        # Training a KMeans model to initialize DMAE
        pretrainer = make_pretrainer()
        pretrainer.fit(X_latent)
        # Obtaining the assigned clusters
        y_pred = pretrainer.predict(X_latent)
        # Pretrain evaluation
        accs1.append(DMAE.Metrics.unsupervised_classification_accuracy(y, y_pred))
        nmis1.append(normalized_mutual_info_score(y, y_pred))
        aris1.append(adjusted_rand_score(y, y_pred))
        
        # Defining DMAE Model
        full_model, full_encoder=make_dmae(encoder_model, decoder_model, X_latent, pretrainer)
        full_model.compile(optimizer=cluster_optimizer["type"](**cluster_optimizer["params"]))
        # Training the full model
        full_model.fit(ds_cluster, **cluster_params)
        # Clustering evaluation
        full_encoder.layers[2].set_weights(full_model.layers[2].get_weights())
        preds = full_encoder.predict(X, steps=test_batch, verbose=False)
        # Obtaning the assigned clusters
        y_pred = np.argmax(preds, axis=1)
        
        # Evaluation
        accs2.append(DMAE.Metrics.unsupervised_classification_accuracy(y, y_pred))
        nmis2.append(normalized_mutual_info_score(y, y_pred))
        aris2.append(adjusted_rand_score(y, y_pred))
        print(f"Trial {i+1}/{trials}\t AE+KMeans: {accs1[-1]} \t AE+DMAE: {accs2[-1]}")
    names = ["Model", "ACC", "NMI", "ARI"]
    print("".join([f'{val:20}' for val in names]))
    print(f"{'AE+KMeans':20}"+"".join(map(lambda i:f'{i:20}',[f'{avg:.4f}'+u"\u00B1"+f'{std:.4f}' for avg, std in zip([np.mean(accs1), np.mean(nmis1), np.mean(aris1)], [np.std(accs1), np.std(nmis1), np.std(aris1)])]))) 
    print(f"{'AE+DMAE':20}"+"".join(map(lambda i:f'{i:20}',[f'{avg:.4f}'+u"\u00B1"+f'{std:.4f}' for avg, std in zip([np.mean(accs2), np.mean(nmis2), np.mean(aris2)], [np.std(accs2), np.std(nmis2), np.std(aris2)])]))) 
