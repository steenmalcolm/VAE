import os
import matplotlib.pyplot as plt
import numpy as np

cur_dir = os.path.dirname(__file__)
def plot_predictions(imgs, imgs_pred):
    plt.figure(figsize=(len(imgs), 2))
    for i in range(len(imgs)):
        plt.subplot(2, len(imgs), i+1)
        plt.imshow(imgs[i].squeeze())
        plt.yticks([])
        plt.xticks([])

        plt.subplot(2, len(imgs), len(imgs)+i+1)
        plt.imshow(imgs_pred[i].numpy().squeeze())
        # plt.imshow(imgs_pred[i].numpy().squeeze())
        plt.yticks([])
        plt.xticks([])
    plt.title('Original Images (Top) vs. AE Output (Bottom)', fontweight='bold')
    plt.savefig(f'{cur_dir}/plots/predictions.png')

def plot_diff(imgs, imgs_pred):
    assert len(imgs)%2==0, 'Use even number of images'
    
    imgs = imgs.squeeze()
    imgs_pred = imgs_pred.numpy().squeeze()

    plt.figure(figsize=(len(imgs)//2, 2))

    for i in range(len(imgs)//2):
        for j in range(2):

            imgs_diff = imgs[j*len(imgs)//2+i]-imgs_pred[j*len(imgs)//2+i]
            min_v, max_v = np.min(imgs_diff), np.max(imgs_diff)
            imgs_diff = (imgs_diff-min_v)/(max_v-min_v)

            plt.subplot(2, len(imgs)//2, j*len(imgs)//2+i+1)
            plt.imshow(imgs_diff)
            plt.yticks([])
            plt.xticks([])
    plt.title('Difference between Input and Output Data', fontweight='bold')
    plt.savefig(f'{cur_dir}/img_diff.png')
