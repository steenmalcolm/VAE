import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner
layers = keras.layers

# Define file paths
cur_dir = os.path.dirname(__file__)

# Prevent overwriting previous trial logs
globals()['trial_count'] = max([
                                int(s.split('_')[1])
                                    for s in os.listdir(f'{os.path.dirname(__file__)}/AE_data/trials')
                                    if 'trial_' in s] or [0])\
                           + 1 or 0
globals()['show_summary'] = True

# Read hyperparameter specs
with open(f'{cur_dir}/hp_specs.json', 'r') as file:
    hp_specs = json.load(file)

L = hp_specs['img_size']
input_shape = (L, L, 1)
num_ellipses = hp_specs['num_ellipses']

def dense_block(x, hp:dict, red_factor:float):
    for _ in range(hp['db_num_layers']):
        temp = x
        x = layers.BatchNormalization()(x)
        bn_features = int(hp['db_num_bn_features']*red_factor+0.5) or 1
        x = layers.Conv2D(
                bn_features,
                (1,1),
                activation=hp['db_activation'],
                kernel_initializer=hp['db_ki'])(x)
        x = layers.BatchNormalization()(x)

        n_features = int(hp['db_growth_rate']*red_factor+0.5) or 1
        x = layers.SeparableConv2D(
                n_features,
                (3, 3),
                padding='same',
                activation=hp['db_activation'],
                kernel_initializer=hp['db_ki'])(x)
        x = layers.Concatenate(axis=-1)([temp, x])
    return x

class InceptionModel(keras_tuner.HyperModel):
    def build(self, hp:keras_tuner.HyperParameters):
        db_hp = dict(
            db_activation=hp.Choice('db_activation', hp_specs['a_funcs']),
            db_ki=hp.Choice('db_ki', hp_specs['initializers']),
            db_growth_rate=hp.Choice('growth_rate', **hp_specs['growth_rate']),
            db_num_layers=hp.Int('db_num_layers', **hp_specs['db_num_layers']),
            db_num_bn_features=hp.Int('db_num_bn_features', **hp_specs['db_num_bn_features']))

 
        num_dense_blocks = hp.Int('num_dense_blocks', **hp_specs['num_dense_blocks'])
        num_bn_features = hp.Int('bottleneck_features', **hp_specs['bottleneck_features'])
        bn_activation = hp.Choice('bottleneck_activation', hp_specs['a_funcs'])
        dense_activation = hp.Choice('dense_activation', hp_specs['a_funcs']+hp_specs['dense_a_funcs'])
        dense_num_units = hp.Int('dense_num_units', **hp_specs['dense_num_units'])
        red_factor = hp.Float('reduction_factor', **hp_specs['reduction_factor'])
        dropout = hp.Float('dropout', **hp_specs['dropout'])

        # Begin model architecture
        input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, strides=(2,2), activation='relu')(input_layer)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPool2D((3, 3), strides=2)


        for i in range(num_dense_blocks):
            x = dense_block(x, db_hp, red_factor**i)
            if i<num_dense_blocks-1:
                if hp.Boolean('max_or_avg_pooling'):
                    x = layers.MaxPool2D((2, 2), padding='same')(x)
                else:
                    x = layers.AvgPool2D((2, 2), padding='same')(x)

            num_red_bn_features = int(num_bn_features*red_factor**i+0.5) or 1
            x = layers.Conv2D(num_red_bn_features, 1, activation=bn_activation)(x)

        encoder_shape = x.shape[1:]
        x = layers.Flatten()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Activation(dense_activation)(x)
        x = layers.Dense(dense_num_units, activation=dense_activation, kernel_initializer='he_uniform')(x)

        # 5 floating points per ellipse for x, y, r1, r2 and angle
        encoded = layers.Dense(5*num_ellipses)(x)

        # Decoder
        x = layers.Activation(dense_activation)(encoded)
        x = layers.Dense(dense_num_units, activation=dense_activation, kernel_initializer='he_uniform')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(np.prod(encoder_shape))(x)
        x = layers.Reshape(encoder_shape)(x)

        for i in range(num_dense_blocks):

            num_red_bn_features = int(num_bn_features*red_factor**(num_dense_blocks-1-i)+0.5) or 1
            x = layers.Conv2D(num_red_bn_features, 1, activation=bn_activation)(x)
            x = dense_block(x, db_hp, red_factor**(num_dense_blocks-1-i))
            if i<num_dense_blocks-1:
                x = layers.UpSampling2D((2, 2))(x)

        decoded = layers.Conv2D(1, 1, padding='same', activation=bn_activation)(x)

        model = keras.models.Model(input_layer, decoded)
        loss_func = hp.Choice('loss', hp_specs['losses'])

        learning_rate = hp.Float('lr', min_value=5e-4, max_value=5e-3, sampling='log')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_func)
        return model


class AutoencoderModel(keras_tuner.HyperModel):
    def build(self, hp:keras_tuner.HyperParameters):
        db_hp = dict(
            db_activation=hp.Choice('db_activation', hp_specs['a_funcs']),
            db_ki=hp.Choice('db_ki', hp_specs['initializers']),
            db_growth_rate=hp.Int('growth_rate', **hp_specs['growth_rate']),
            db_num_layers=hp.Int('db_num_layers', **hp_specs['db_num_layers']),
            db_num_bn_features=hp.Int('db_num_bn_features', **hp_specs['db_num_bn_features']))

 
        num_dense_blocks = hp.Int('num_dense_blocks', **hp_specs['num_dense_blocks'])
        num_bn_features = hp.Int('bottleneck_features', **hp_specs['bottleneck_features'])
        bn_activation = hp.Choice('bottleneck_activation', hp_specs['a_funcs'])
        dense_activation = hp.Choice('dense_activation', hp_specs['a_funcs']+hp_specs['dense_a_funcs'])
        dense_num_units = hp.Int('dense_num_units', **hp_specs['dense_num_units'])
        red_factor = hp.Float('reduction_factor', **hp_specs['reduction_factor'])
        dropout = hp.Float('dropout', **hp_specs['dropout'])

        # Begin model architecture
        input_layer = layers.Input(shape=input_shape)
        x = layers.Conv2D(num_bn_features, 1, activation=bn_activation)(input_layer)
        for i in range(num_dense_blocks):
            x = dense_block(x, db_hp, red_factor**i)
            if i<num_dense_blocks-1:
                if hp.Boolean('max_or_avg_pooling'):
                    x = layers.MaxPool2D((2, 2), padding='same')(x)
                else:
                    x = layers.AvgPool2D((2, 2), padding='same')(x)

            num_red_bn_features = int(num_bn_features*red_factor**i+0.5) or 1
            x = layers.Conv2D(num_red_bn_features, 1, activation=bn_activation)(x)

        encoder_shape = x.shape[1:]
        x = layers.Flatten()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Activation(dense_activation)(x)
        x = layers.Dense(dense_num_units, activation=dense_activation, kernel_initializer='he_uniform')(x)

        # 5 floating points per ellipse for x, y, r1, r2 and angle
        encoded = layers.Dense(5*num_ellipses)(x)

        # Decoder
        x = layers.Activation(dense_activation)(encoded)
        x = layers.Dense(dense_num_units, activation=dense_activation, kernel_initializer='he_uniform')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(np.prod(encoder_shape))(x)
        x = layers.Reshape(encoder_shape)(x)

        for i in range(num_dense_blocks):

            num_red_bn_features = int(num_bn_features*red_factor**(num_dense_blocks-1-i)+0.5) or 1
            x = layers.Conv2D(num_red_bn_features, 1, activation=bn_activation)(x)
            x = dense_block(x, db_hp, red_factor**(num_dense_blocks-1-i))
            if i<num_dense_blocks-1:
                x = layers.UpSampling2D((2, 2))(x)

        decoded = layers.Conv2D(1, 1, padding='same', activation=bn_activation)(x)

        model = keras.models.Model(input_layer, decoded)
        loss_func = hp.Choice('loss', hp_specs['losses'])

        learning_rate = hp.Float('lr', min_value=5e-4, max_value=5e-3, sampling='log')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_func)
        return model

    def fit(self, hp:keras_tuner.HyperParameters, model:keras.models.Model, *args, **kwargs):

        print(f'Training model with {model.count_params()} params')
        batch_size=hp.Int('batch_size', 32, 64, step=2, sampling='log')

        globals()['trial_count']+=1
        trial_count = globals()['trial_count']

        if globals()['show_summary']:
            globals()['show_summary'] = False
            print(model.summary())
        kwargs['callbacks'] += [keras.callbacks.CSVLogger(f'AE_logs/history/{trial_count:d}.csv', append=False)]
        return model.fit(*args, batch_size=batch_size, **kwargs)

class DecoderModel(keras_tuner.HyperModel):
    def build(self, hp:keras_tuner.HyperParameters):

        activation = hp.Choice('dense_activation', hp_specs['a_funcs']+hp_specs['dense_a_funcs'])
        dropout = hp.Float('dropout', 0.005, 0.5, sampling='log')

        input_layer = layers.Input(shape=(3,))
        x = input_layer
        for i in range(hp.Int('num_dense_layers', 1, 10)):
            x = layers.Dense(hp.Int(f'units_layer{i}', 4, 128, step=2, sampling='log'), activation=activation)(x)
            x = layers.Dropout(dropout)
        if hp.Boolean('shortcut'):
            x = layers.Concatenate()([x, input_layer])
        x = layers.BatchNormalization()(x)
        x = layers.Dense(16, activaton='relu')(x)

        decoded = layers.Dense(3, activation='linear')

        model = keras.models.Model(input_layer, decoded)
        loss_func = hp.Choice('loss', hp_specs['losses'])

        learning_rate = hp.Float('lr', min_value=5e-4, max_value=1e-2, sampling='log')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_func)

        return model
    
    def fit(self, hp:keras_tuner.HyperParameters, model:keras.models.Model, *args, **kwargs):
    
        print(f'Training model with {model.count_params()}')
        batch_size=hp.Choice('batch_size', [32, 64, 128])
        if model.count_params()>1e5:
            kwargs['epochs'] = 1
        return model.fit(*args, batch_size=batch_size, **kwargs)
