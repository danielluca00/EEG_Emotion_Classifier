import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def create_dnn(input_dim, num_classes=3):
    i_dnn = Input(shape=(input_dim,))

    x = Dense(2548, activation='relu')(i_dnn)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Dense(3822, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.27)(x)

    x = Dense(5096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(3822, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.27)(x)

    x = Dense(2548, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    y = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=i_dnn, outputs=y)
    return model
