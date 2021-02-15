from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from deepdmae import DeepDMAE
import utils

def make_models(
        encoder_params,
        decoder_params,
        dmae_params,
        loss_params,
        input_shape
        ):
    #TODOC
    inp = Input(shape=input_shape)
    model = DeepDMAE(
            encoder_params,
            decoder_params,
            dmae_params
            )

    X_tilde_pre = model.autoencode(inp)
    X_tilde, H, theta_tilde = model(inp)
    assigns = model.assign(inp)

    autoencoder = Model(
            inputs=[inp],
            outputs=[X_tilde_pre]
            )

    compile_params = utils.optimizer_lut(
            loss_params["autoencoder"]
            )

    autoencoder.compile(
            **compile_params
            )

    full_model = Model(
            inputs=[inp],
            outputs=[X_tilde]
            )
    
    full_model.add_loss(
            utils.dmae_loss(
                inp, X_tilde, H, 
                theta_tilde, dmae_params, 
                loss_params
                )
            )
    compile_params = utils.optimizer_lut(loss_params["dmae"])
    full_model.compile(optimizer=compile_params["optimizer"])

    encoder_model = Model(
            inputs=[inp],
            outputs=[H]
            )

    assign_model = Model(
            inputs=[inp],
            outputs=[assigns]
            )

    return {
            "autoencoder": autoencoder,
            "full_model": full_model,
            "encoder": encoder_model,
            "assign_model": assign_model
            }
