import gym
import ipdb
from PIL import Image
from skimage.transform import resize
#from skimage.io import imshow
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scipy.misc

env = gym.make("CartPole-v0")

state = env.reset()

img = env.render(mode='rgb_array')

img_resized = resize(img, (img.shape[0] / 4, img.shape[1] / 4), anti_aliasing=True)

t= plt.imshow(img_resized)

ipdb.set_trace()



