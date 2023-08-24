import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from gen_data import gen_ellipse_data
from tensorflow import keras
from build_model import AutoencoderModel

import keras_tuner
layers = keras.layers
cur_dir = os.path.dirname(__file__)

def main():

    data_path = f'{cur_dir}/AE_data/training_data/data.pkl'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as file:
            data = pickle.load(file)

    else:
        data = gen_ellipse_data(10010)
        with open(data_path, 'wb') as file:
            pickle.dump(data.tolist(), file)

    train_data, test_data = np.expand_dims(data[:-10], axis=-1), np.expand_dims(data[-10:], axis=-1)

    tuner = keras_tuner.RandomSearch(
        AutoencoderModel(),
        objective='val_loss',
        max_trials=20,
        executions_per_trial=1,
        overwrite=True,
        directory=f'{cur_dir}/AE_data/trials',
        project_name='some_trial'
    )

    train_args = dict(
        x=train_data,
        y=train_data,
        epochs=20,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.67, patience=4, verbose=1)])

    tuner.search(**train_args)

    best_models = tuner.get_best_models(num_models=2)
    for i, model in enumerate(best_models):
        model.save(f'{cur_dir}/models/{i}.h5')
    
    
    # model.fit(**train_args)
    # model.save('model.h5')
    
    # Decode x, y coordinates and radius
    # else:
    #     model:keras.Model = keras.models.load_model('model.h5')
    #     encoder = keras.models.Model(model.input, model.get_layer(name='dense').output)

    #     params_decoder = keras.models.Sequential(layers=[
    #         layers.Dense(124, input_shape=(3,), activation='relu'),
    #         layers.Dense(124, activation='relu'),
    #         layers.Dense(3, activation='relu')])
        
    #     params_decoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')

    #     ndata = 10000
    #     encoder_input, output = gen_encoder_data(10000)
    #     data = []
    #     for i in range(5):
    #         data+= encoder(encoder_input[i*ndata//5:(i+1)*ndata//5]).numpy().tolist()
    #     data = np.array(data)
    #     train_data, test_data = data[:-ndata//5], data[-ndata//5:]
    #     train_output, test_output = output[:-ndata//5], output[-ndata//5:]

    #     train_args = dict(
    #         x=train_data,
    #         y=train_output,
    #         batch_size=128,
    #         epochs=100,
    #         validation_split=0.1,
    #         callbacks=[
    #             keras.callbacks.CSVLogger('history.csv', append=True),
    #             keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    #             keras.callbacks.ReduceLROnPlateau(factor=0.67, patience=10, verbose=1)])

    #     params_decoder = keras.models.load_model('params_decoder.h5')
    #     params_decoder.fit(**train_args)
    #     params_decoder.save('params_decoder.h5')
        
        
if __name__ == '__main__':
    main()
