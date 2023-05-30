import os
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from fast_slic import Slic
from PIL import Image
import pdb
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import random

imgs = glob.glob('tinyImagenet/tiny-imagenet-200/train/*/*/*.JPEG')
selected_imgs = random.choices(imgs, k=300)
for idx, img in enumerate(selected_imgs):
# for idx, img in tqdm(enumerate(selected_imgs)):
    with Image.open(img) as f:
        image = np.array(f.resize((224,224)))
    numSegments = 250
    fastslic = Slic(num_components=numSegments, compactness=10)
    segments = slic(image, n_segments = numSegments, sigma = 5)
    assignment = fastslic.iterate(image) # Cluster Map
    if segments.max() != 169: print('idx: '+str(idx)+', scipy slic error: '+str(segments.max()))
    if segments.max() != 169: print('idx: '+str(idx)+', fast slic error: '+str(assignment.max()))
    pdb.set_trace()
    continue
    pdb.set_trace()
    
    marked_img = mark_boundaries(image, assignment)
    plt.imsave((os.path.join('output/visualization', os.path.split(img)[1])), image)
    plt.imsave((os.path.join('output/visualization', 'marked_'+os.path.split(img)[1])), marked_img)