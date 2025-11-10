import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def create_dnn(input_dim, num_classes=3, dropout_rates=[0.3, 0.3, 0.25]):
    i = Input(shape=(input_dim,))

    x = Dense(512, activation='relu')(i)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rates[0])(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rates[1])(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rates[2])(x)

    y = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=i, outputs=y)
    return model
