import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import keras_tuner
from gen_data import gen_ellipse_data
layers = keras.layers
keras.models.Model().save
layers.BatchNormalization()

def inception_module(
        x,
        filters_1x1,
        filters_3x3_reduce,
        filters_3x3,
        filters_5x5_reduce,
        filters_5x5,
        filters_pool_proj,
        name_pre):
    
    conv_kwargs = dict(
        padding='same',
        activation='selu',
        bias_initializer=keras.initializers.Constant(0.2))

    conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), **conv_kwargs, name=f'{name_pre}_1x1')(x)
 
    # conv_3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), **conv_kwargs, name=f'{name_pre}_3x3_reduce')(x)
    # conv_3x3 = layers.SeparableConv2D(filters_3x3, (3, 3), **conv_kwargs, name=f'{name_pre}_3x3')(conv_3x3)
    conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), **conv_kwargs, name=f'{name_pre}_3x3')(x)

    # conv_5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), **conv_kwargs, name=f'{name_pre}_5x5_reduce')(x)
    # conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), **conv_kwargs, name=f'{name_pre}_5x5')(conv_5x5)
    conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), **conv_kwargs, name=f'{name_pre}_5x5')(x)

    pool_proj = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same', name=f'{name_pre}_pool')(x)
    pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), **conv_kwargs, name=f'{name_pre}_pool_1x1')(pool_proj)

    output = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name_pre)
    output = layers.BatchNormalization()(output)
 
    return output

def visualize_output(model, num_test_imgs=10):
    test_imgs = gen_ellipse_data(num_test_imgs)
    test_imgs = np.expand_dims(test_imgs, axis=-1)
    if isinstance(model, str):
        model = keras.models.load_model(model)
    predictions = model(test_imgs).numpy()
    plt.figure(figsize=(len(test_imgs),2))
    plt.title('Original Images (Top) vs. AE Output (Bottom)', fontweight='bold')
    for i in range(len(test_imgs)):
        test_imgs[i,:3,:3]=0
        predictions[i,:3,:3]=0
        plt.subplot(2, len(test_imgs), i+1)
        loss = np.mean(np.sqrt((test_imgs[i]-predictions[i])**2))
        plt.title(f'{loss*100:.1f}')
        plt.imshow(test_imgs[i].squeeze())
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, len(test_imgs), len(test_imgs)+i+1)
        plt.imshow(predictions[i].squeeze())
        plt.yticks([])
        plt.xticks([])

    plt.savefig('predictions.png')
    plt.close()



def build_inception_model(input_shape=(128,128,1)):
    input_layer = layers.Input(shape=input_shape)
    layer_kwargs = dict(padding='same', activation='selu', kernel_initializer='he_uniform', bias_initializer=keras.initializers.Constant(0.2))
    x = layers.Conv2D(64, 5, **layer_kwargs)(input_layer)
    x = layers.Conv2D(64, 3, strides=(2,2), **layer_kwargs, name='shrink_1')(x)
    x = layers.Conv2D(32, 3, **layer_kwargs)(x)
    x = layers.Conv2D(32, 3, strides=(2,2), **layer_kwargs, name='shrink_2')(x)
    x = inception_module(x, 8, 8, 12, 4, 8, 4, 'enc_inc_1')
    x = layers.Conv2D(16, 3, strides=(2,2), **layer_kwargs, name='shrink_3')(x)
    x = inception_module(x, 4, 6, 4, 4, 4, 4, 'enc_inc_2')
    x = layers.Conv2D(16, 3, strides=(2,2), **layer_kwargs, name='shrink_4')(x)
    x = inception_module(x, 2, 2, 2, 2, 2, 2, 'enc_inc_3')
    x = layers.Conv2D(8, 3, strides=(2,2), **layer_kwargs, name='shrink_5')(x)
    encoder_shape = x.shape[1:]
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='selu', bias_initializer=keras.initializers.Constant(0.2))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='selu', bias_initializer=keras.initializers.Constant(0.2))(x)
    x = layers.Dropout(0.3)(x)
    encoded = layers.Dense(20, bias_initializer=keras.initializers.Constant(0.2))(x)

    x = layers.Activation('linear')(encoded)
    x = layers.Dense(32, activation='selu', bias_initializer=keras.initializers.Constant(0.2))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='selu', bias_initializer=keras.initializers.Constant(0.2))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(np.prod(encoder_shape))(x)
    x = layers.Reshape(encoder_shape)(x)
    x = layers.Conv2DTranspose(8, 3, strides=(2,2), **layer_kwargs, name='expand_1')(x)
    x = inception_module(x, 4, 4, 4, 4, 4, 4, 'dec_inc_1')
    x = layers.Conv2DTranspose(16, 3, strides=(2,2), **layer_kwargs, name='expand_2')(x)
    x = inception_module(x, 4, 4, 4, 4, 4, 4, 'dec_inc_2')
    x = layers.Conv2DTranspose(32, 3, strides=(2,2), **layer_kwargs, name='expand_3')(x)
    x = inception_module(x, 8, 8, 12, 4, 8, 4, 'dec_inc_3')
    x = layers.Conv2DTranspose(32, 3, strides=(2,2), **layer_kwargs, name='expand_4')(x)
    x = layers.Conv2DTranspose(64, 3, **layer_kwargs)(x)
    x = layers.Conv2DTranspose(64, 3, strides=(2,2), **layer_kwargs, name='expand_5')(x)
    decoded = layers.Conv2DTranspose(1, 5, activation='sigmoid', padding='same', bias_initializer=keras.initializers.Constant(0.2))(x)

    model = keras.models.Model(input_layer, decoded)
    model.compile(keras.optimizers.Adam(5e-4), 'mae')
    return model

def visualize_layers(model:keras.models.Model):
    test_img = gen_ellipse_data(1)
    plt.imshow(test_img[0])
    plt.savefig('layer_activations/original.png')
    plt.close()
    prediction = model(test_img)[0]
    plt.imshow(prediction)
    plt.savefig('layer_activations/output.png')
    plt.close()
    count = 0
    for layers in model.layers:
        if 'shrink' in layers.name or 'expand' in layers.name:
            count+=1
            if 'shrink' in layers.name:
                model_test = keras.models.Model(model.input, layers.output)
            else:
                model_test = keras.models.Model(model.input, layers.input)
            layer_activation = model_test(test_img)[0]
            num_filters = layer_activation.shape[-1]
            plt.figure(figsize=(num_filters//8,8))
            for filter_id in range(num_filters):
                plt.subplot(8, num_filters//8, filter_id+1)
                plt.imshow(layer_activation[:,:,filter_id].numpy().squeeze())
                plt.xticks([])
                plt.yticks([])
            plt.savefig(f'layer_activations/{layers.name}.png')
            plt.close()


def generate_max_activation_image(model:keras.models.Model, iterations=200, step=1.0):
    input_shape = model.input_shape[1:]
    for layer in model.layers:
        if 'shrink' in layer.name or 'expand' in layer.name:
            if 'shrink_' in layer.name:
                activation_model = keras.models.Model(model.input, layer.output)

            elif 'expand_' in layer.name:
                activation_model = keras.models.Model(model.input, layer.input)
            

            # Create a random image to start with

            # Perform gradient ascent
            num_filters = layer.output_shape[-1]
            plt.figure(figsize=(num_filters//8, 8))
            for filter_id in range(num_filters):
                input_img_data = tf.convert_to_tensor(np.random.random((1,)+input_shape))
                print(layer.name, filter_id)
                for i in range(iterations):
                    with tf.GradientTape() as tape:
                        tape.watch(input_img_data)
                        output = activation_model(input_img_data)
                        # output = tf.expand_dims(output[..., filter_id], axis=-1)
                        loss = tf.reduce_mean(output[..., filter_id])

                    # Compute the gradients of the loss with respect to the input image
                    grads = tape.gradient(loss, input_img_data)

                    # Normalize the gradients
                    grads /= (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

                    # Update the image
                    input_img_data += grads * step

                plt.subplot(8, num_filters//8, filter_id+1)
                plt.imshow(input_img_data.numpy().squeeze())
                plt.xticks([])
                plt.yticks([])
            plt.savefig(f'layer_activations/max_activation/{layer.name}.png')
            plt.close()

# model = build_inception_model()
# print(model.summary())

model = keras.models.load_model('induct.h5')
# visualize_layers(model)
# generate_max_activation_image(model)

for i in range(100):
    print(i)
    data = gen_ellipse_data(1000)
    train_data = np.expand_dims(data, axis=-1)

    train_args = dict(
        x=train_data,
        y=train_data,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.CSVLogger('history.csv'),
            keras.callbacks.EarlyStopping(patience=7, min_delta=0.0001, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.67, patience=4, verbose=1)])
 
    model.fit(**train_args)
    model.save('induct.h5')
visualize_output(model)
visualize_layers(model)
