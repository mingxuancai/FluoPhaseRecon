import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

def showimgstack(img, num):
    f, axes = plt.subplots(1, 1, figsize=(10, 6))
    frames = []

    frames.append(axes.imshow(img[0]))
    axes.set_title('filled in phantom')

    interact(frames[0].set_data(img[i]), i=IntSlider(min=0, max=num-1, step=1, value=0))