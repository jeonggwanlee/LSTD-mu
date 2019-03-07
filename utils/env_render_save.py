import gym
import ipdb
from PIL import Image
from skimage.transform import resize
#from skimage.io import imshow
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scipy.misc
from skimage.util import crop

env = gym.make("CartPole-v0")

state = env.reset()
for i in range(30):
    img = env.render(mode='rgb_array')
    img2 = img[300:700]
    img_resized = resize(img2, (img2.shape[0]/16, img2.shape[1]/16), anti_aliasing=True)
    scipy.misc.imsave("outfile_%d.jpg" % (i), img_resized)
    action = 1
    next_state, reward, done, info = env.step(action)
    state = next_state
    if done:
        ipdb.set_trace()
        break
    print(i)
    print(reward)
    ipdb.set_trace()

