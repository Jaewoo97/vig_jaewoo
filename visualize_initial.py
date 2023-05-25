import os
import numpy as np
from skimage.segmentation import mark_boundaries

from fast_slic import Slic
from PIL import Image
import pdb
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import random

imgs = glob.glob('tinyImagenet/tiny-imagenet-200/train/*/*/*.JPEG')
selected_imgs = random.choices(imgs, k=30)
for img in tqdm(selected_imgs):
    with Image.open(img) as f:
        image = np.array(f.resize((224,224)))
    numSegments = 100
    slic = Slic(num_components=numSegments, compactness=10)
    try:
        assignment = slic.iterate(image) # Cluster Map
    except:
        pdb.set_trace()
    pdb.set_trace()
    marked_img = mark_boundaries(image, assignment)
    plt.imsave((os.path.join('output/visualization', os.path.split(img)[1])), image)
    plt.imsave((os.path.join('output/visualization', 'marked_'+os.path.split(img)[1])), marked_img)