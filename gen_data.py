import numpy as np
import matplotlib.pyplot as plt

def gen_ellipse_data(num_data: int, num_ellipses:int):
    '''Generate 64x64 images with ellipses'''
    M = 32
    imgs = np.zeros((num_data, M, M), dtype=int)

    for i in range(num_data):
        ellipse_count = 0
        if i%1000==0:
            print(i)
        while ellipse_count<num_ellipses:

            img = np.zeros((M, M), dtype=int)
            x, y = np.random.randint(M//8+2, 7*M//8-2, 2)
            max_r = min([x, M-x, y, M-y, M//4])-1
            try:
                r1, r2 = np.random.randint(M//8-1, max_r, 2)
            except ValueError as e:
                print(x,y,M//8, max_r)
                exit(0)
            angle = np.random.random()*2*np.pi


            c_a, s_a = np.cos(angle), np.sin(angle)
            c_a2, s_a2 =  np.cos(angle)**2, np.sin(angle)**2
            r1_2, r2_2 = r1**2, r2**2
            max_y = int(np.sqrt(r1_2*s_a2 + r2_2*c_a2)+0.5)

            for yi in range(-max_y, max_y):
 
                x_range = [int( 1/(c_a2*r2_2+s_a2*r1_2) * \
                            (yi*s_a*c_a*(r1_2-r2_2) + \
                            sign*r1*r2*np.sqrt(max_y**2 - yi**2)) + 0.5)\
                        for sign in [-1, 1]]

                for xi in range(*x_range):
                    img[x+xi][y+yi] = 1.
            if not (img+imgs[i]>1).any():
                imgs[i] += img
                ellipse_count += 1
    return imgs

def gen_sphere_data(num_data:int, length:int):
    '''Generate ```num_data``` data points for autoencoder with dimension length x length'''

    x_vals, y_vals = np.random.randint(length//8, 7*length//8, 2*num_data).reshape(2, num_data)

    data = np.zeros((num_data, length, length), dtype=bool)
    for i in range(num_data):
        x, y = x_vals[i], y_vals[i]
        max_r = min([x, length-x, y, length-y])-1
        r = np.random.randint(min([max_r-1, 10]), max_r)
        
        for xi in range(x-r, x+r):
            r_x = int(np.sqrt(r**2 - (x-xi)**2))
            for yi in range(y-r_x, y+r_x):
                data[i, xi, yi] = True

    return data.tolist()


def gen_encoder_data(num_data: int, length:int):
    '''Generate data for training DNN to map output of encoder to x, y and r values'''

    x_vals, y_vals = np.random.randint(length//8, 7*length//8, 2*num_data).reshape(2, num_data)
    r_vals = np.empty(num_data, dtype=int)

    data = np.zeros((num_data, length, length, 1), dtype=float)
    for i in range(num_data):
        x, y = x_vals[i], y_vals[i]
        max_r = min([x, length-x, y, length-y])-1
        r_vals[i] = np.random.randint(min([max_r-1, 10]), max_r)
        r = r_vals[i]

        for xi in range(x-r, x+r):
            r_x = int(np.sqrt(r**2 - (x-xi)**2))
            for yi in range(y-r_x, y+r_x):
                data[i, xi, yi, 0] = True

    return data, np.stack((x_vals, y_vals, r_vals), axis=-1)



if __name__ == '__main__':
    from visualize_results import plot_predictions
    imgs1 = gen_ellipse_data(10)
    imgs2= gen_ellipse_data(10)
    plot_predictions(imgs1, imgs2)