from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import sys

name= sys.argv[1]
output= sys.argv[2]
model=load_model(name)
r, c = 5, 5
noise = np.random.normal(0, 1, (r * c, 100))
gen_imgs = model.predict(noise)
# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
fig.savefig(output)         


