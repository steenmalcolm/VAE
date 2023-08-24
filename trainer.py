import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import keras_tuner
from gen_data import gen_ellipse_data
from build_model import AutoencoderModel
cur_dir = os.path.dirname(__file__)
keras.models.load_

def get_best_hp()->dict:
    trials_path = f'{cur_dir}/AE_logs/trials'
    # trial_files = os.listdir(trials_path)
    trial_files = ['.']
    best_hp = {}
    best_score = 1.
    for trial_file in trial_files:
        # if 'trial_' in trial_file:
        if True:
            # with open(os.path.join(trials_path, trial_file, 'trial.json'), 'r') as fp:
            with open('trial.json', 'r') as fp:
                hp_dict = json.load(fp)
                print(list(hp_dict['hyperparameters']['values'].values()))
            score = hp_dict['score'] 
            if score < best_score:
                best_score = score
                best_hp = hp_dict['hyperparameters']['values']
                
    return best_hp

def train_best_model():
    
    data = gen_ellipse_data(10010)
    train_data, test_data = np.expand_dims(data[:-10], axis=-1), np.expand_dims(data[-10:], axis=-1)

    train_args = dict(
        x=train_data,
        y=train_data,
        epochs=20,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.67, patience=4, verbose=1)])

   
    best_hp = get_best_hp()
    hp = keras_tuner.HyperParameters()
    hp.values = best_hp
    model = AutoencoderModel().build(hp)

    model.fit(batch_size=hp['batch_size'], **train_args)
    model.save(f'{cur_dir}/AE_logs/models/trained_model.h5')

def visualize_output(model, test_imgs):
    if isinstance(model, str):
        model = keras.models.load_model(f'{cur_dir}/AE_logs/models/{model}')
    predictions = model(test_imgs)
    plt.figure(figsize=(len(test_imgs),2))
    for i in range(len(test_imgs)):
        plt.subplot(2, len(test_imgs), i+1)
        plt.imshow(test_imgs[i].squeeze())
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, len(test_imgs), len(test_imgs)+i+1)
        plt.imshow(predictions[i].numpy().squeeze())
        plt.yticks([])
        plt.xticks([])
    plt.title('Original Images (Top) vs. AE Output (Bottom)', fontweight='bold')
    plt.savefig(f'{cur_dir}/AE_logs/plots/predictions.png')

