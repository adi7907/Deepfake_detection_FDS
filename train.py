# MesoNet Deepfake Detection Training Script
# This script contains the standalone architecture and training pipeline.

import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_meso4_model(input_shape=(128, 128, 3)):
    x = Input(shape=input_shape)
    x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    
    x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
    
    x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
    
    x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
    
    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)
    
    model = Model(inputs=x, outputs=y, name='Meso4_Deepfake_Detector')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = create_meso4_model()
    model.summary()
    print('Model architecture loaded successfully.')
