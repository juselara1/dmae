from tensorflow.keras import Model as _Model

from utils import layers_lut as _layers_lut, dissimilarities_lut as _dissimilarities_lut 
from utils import dmae_lut as _dmae_lut, initializer_lut as _initializer_lut

class StackedLayers(_Model):
    #TODOC
    def __init__(
            self,
            params = None,
            **kwargs
            ):
        
        super(StackedLayers, self).__init__(**kwargs)
        self.__params = params
        self.__make_layers()

    def __make_layers(self):
        #TODOC
        self.__layers = []
        for param in self.__params:
            if param["kind"] in ["dense", "conv"]:
                param = _initializer_lut(param)
            layer = _layers_lut(param)
            self.__layers.append(
                    layer
                    )
    
    def call(self, input_tensor):
        #TODOC
        layer = input_tensor
        for lay in self.__layers:
            layer = lay(layer)
        return layer

class DeepDMAE(_Model):
    #TODOC
    def __init__(
            self,
            encoder_params = None,
            decoder_params = None,
            dmae_params = None,
            **kwargs
            ):
        super(DeepDMAE, self).__init__(**kwargs)

        self.__encoder = StackedLayers(encoder_params)
        self.__decoder = StackedLayers(decoder_params)
        dmae_params = _dissimilarities_lut(dmae_params)
        self.__dmae, self.__dmaen = _dmae_lut(dmae_params)

    def call(self, X):
        #TODOC
        H = self.__encoder(X)
        theta_tilde = self.__dmae(H)
        X_tilde = self.__decoder(theta_tilde[0])
        return X_tilde, H, theta_tilde

    def autoencode(self, X):
        #TODOC
        H = self.__encoder(X)
        X_tilde = self.__decoder(H)
        return X_tilde

    def assign(self, X):
        #TODOC
        H = self.__encoder(X)
        assigns = self.__dmaen(H)
        return assigns
