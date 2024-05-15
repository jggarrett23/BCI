# EEGNet-specific imports
from arl_eegmodels.EEGModels import EEGNet

import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC
import tensorflow_addons as tfa

class Hyper_EEGNet(kt.HyperModel):
    
    """Create HyperModel class of EEGNet for hyperparamter tuning """
    def __init__(self, nClasses, nSamples, nChans, kernLength, **kwargs):
        super(Hyper_EEGNet, self).__init__()
        
        # necessary paramaters to construct model that are not being tuned
        self.nb_classes = nClasses
        self.nSamples = nSamples
        self.nChans = nChans
        self.kernLength = kernLength
        self.dropoutRate = kwargs.get('dropoutRate', 0.5)
        self.dropoutType = kwargs.get('dropoutType', 'Dropout')
        
        
    # create model
    def build(self, hp):
        
        opt_F1 = hp.Int(name='F1', min_value=8, max_value=24, step=8)
        opt_D = hp.Int(name='D', min_value=2, max_value=6, step=2)
        opt_F2 = hp.Int(name='F2', min_value=16, max_value=144, step=16)
        opt_lr = hp.Choice(name='learning_rate', values=[1e-2, 1e-3, 1e-4])
    
        model = EEGNet(nb_classes = self.nb_classes, Chans = self.nChans, Samples = self.nSamples, 
                       dropoutRate = self.dropoutRate, kernLength = self.kernLength, 
                       F1 = opt_F1, D = opt_D, F2 = opt_F2, dropoutType = self.dropoutType)
    
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = opt_lr), 
                      metrics=['accuracy', 
                               AUC(name='auc'),
                              tfa.metrics.F1Score(num_classes=self.nb_classes, average='macro', name='f1_score')
                              ])
        
        return model
    
