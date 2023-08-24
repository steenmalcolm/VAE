import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.initializers import Constant
from keras import backend as K
import numpy as np
from visualize_results import plot_predictions
from gen_data import gen_ellipse_data


num_ellipses = 3
dof = 5 # Degrees of freedome

train_data = gen_ellipse_data(20000, num_ellipses)
train_data = np.expand_dims(train_data, axis=-1)
input_shape = train_data.shape[1:]

conv_layer_args = dict(padding='same', activation='relu', bias_initializer=Constant(0.2))
input_img = Input(shape=input_shape, name='encoder_input')
x = Conv2D(16, 3,  **conv_layer_args)(input_img)
x = Conv2D(32, 3, **conv_layer_args, strides=(2,2))(x)
x = Conv2D(32, 3, **conv_layer_args)(x)
x = Conv2D(32, 3, **conv_layer_args)(x)
conv_shape = K.int_shape(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
encoder_output = Dense(num_ellipses*dof, activation='relu', name='encoder_output')(x)


x = Dense(512, activation='relu')(encoder_output)
x = Dropout(0.3)(x)
x = Dense(np.prod(conv_shape[1:]), activation='relu')(x)
x = Dropout(0.3)(x)
x = Reshape(conv_shape[1:])(x)
x = Conv2DTranspose(32, 3, **conv_layer_args, strides=(2,2))(x)
x = Conv2DTranspose(32, 3, **conv_layer_args)(x)
decoder_output = Conv2DTranspose(1, 3, **conv_layer_args, name='decoder_output')(x)
model = Model(input_img, decoder_output)

train_args = dict(
    x=train_data,
    y=train_data,
    epochs=25,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        keras.callbacks.CSVLogger(f'history.csv', append=False),
        keras.callbacks.ReduceLROnPlateau(factor=0.67, patience=4, verbose=1)])


model = keras.models.load_model('test_ae.h5')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6), loss='mse')
model.fit(**train_args)
model.save('test_ae.h5')
test_data = gen_ellipse_data(10, num_ellipses)
test_data = np.expand_dims(test_data, axis=-1)
test_predictions = model(test_data)
plot_predictions(test_data, test_predictions)

