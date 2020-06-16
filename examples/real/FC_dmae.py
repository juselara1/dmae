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
              train_batch, test_batch, dis, dis_loss, pretrainer, input_shape,
              cov, X_latent):
    # Defining DMAE Model
    inp = tf.keras.layers.Input(shape=(latent_dim, ))
    # DMAE layer
    if cov:
        DMAE_layer = DMAE.Layers.DissimilarityMixtureAutoencoderCov(10000, n_clusters, dissimilarity=dis,
                                                                    trainable={"centers":True, "cov":True, "mixers":True},
                                                                    batch_size=train_batch, grad_modifier=1,
                                                                    initializers=init_dmae(pretrainer, X_latent))(inp)
    else:
        DMAE_layer = DMAE.Layers.DissimilarityMixtureAutoencoder(10000, n_clusters, dissimilarity=dis,
                                                                 trainable={"centers":True, "mixers":True},
                                                                 batch_size=train_batch, 
                                                                 initializers=init_dmae(pretrainer))(inp)
    # defining keras model
    DMAE_model = tf.keras.Model(inputs=[inp], outputs=[DMAE_layer])
    # Defining complete model
    model_in = tf.keras.layers.Input(shape=input_shape)
    latent = encoder_model(model_in)
    dmae_rec = DMAE_model(latent)
    if cov:
        full_rec = decoder_model(dmae_rec[0][0])
        full_model = tf.keras.Model(inputs=[model_in], outputs=[full_rec])
        # Loss function and optimizer
        loss = 0.01*tf.losses.mse(model_in, full_rec)+1.0*dis_loss(latent, *dmae_rec[0])
    else:
        full_rec = decoder_model(dmae_rec)
        full_model = tf.keras.Model(inputs=[model_in], outputs=[full_rec])
        # Loss function and optimizer
        loss = 0.01*tf.losses.mse(model_in, full_rec)+1.0*dis_loss(latent, dmae_rec)
    full_model.add_loss(loss)
    # Defining a deep encoder
    if cov:
        DMAE_encoder = DMAE.Layers.DissimilarityMixtureEncoderCov(10000, n_clusters, dissimilarity=dis,
                                                                  batch_size=test_batch)(latent)
    else:
        DMAE_encoder = DMAE.Layers.DissimilarityMixtureEncoder(10000, n_clusters, dissimilarity=dis,
                                                               batch_size=test_batch)(latent)
    full_encoder = tf.keras.Model(inputs=model_in, outputs=DMAE_encoder)
    return full_model, full_encoder

def train(ds_pretrain, ds_cluster, ds_test, y, N, test_batch, trials, make_autoencoder, make_dmae,
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
        ae_model.fit(ds_pretrain, **pretrain_params)
        # Data representation in the latent space
        X_latent = encoder_model.predict(ds_test, steps=N//test_batch)
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
        full_model, full_encoder = make_dmae(encoder_model, decoder_model, X_latent, pretrainer)
        full_model.compile(optimizer=cluster_optimizer["type"](**cluster_optimizer["params"]))
        # Training the full model
        callbacks = [DMAE.Callbacks.DeltaUACC(full_encoder, ds_test, N, verbose=False, interval=140, batch_size=test_batch, tol=1e-4)]
        full_model.fit(ds_cluster, callbacks=callbacks, **cluster_params)
        # Clustering evaluation
        preds = full_encoder.predict(ds_test, steps=N//test_batch)
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