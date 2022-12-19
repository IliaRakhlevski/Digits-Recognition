# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:06:29 2018

@author: Iliar
"""

import numpy as np
from skimage import io
import scipy.signal as sig
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from functools import reduce

# edges
G_x = 0
G_y = 0

N_BUCKETS = 18      # number of buckets in histogram
CELL_SIZE = 8       # Each cell is 8x8 pixels
BLOCK_SIZE = 2      # Each block is 2x2 cells


def assign_bucket_vals(m, d, bucket_vals):
    if d >= 20 * N_BUCKETS:
        d = d - (20 * N_BUCKETS)
        
    right_bin = int(d / 20.)      
    left_bin = (int(d / 20.) + 1)
    
    right_val= m * (left_bin * 20 - d) / 20
    left_val = m * (d - right_bin * 20) / 20
    
    if left_bin == N_BUCKETS:
        left_bin = 0
    
    bucket_vals[right_bin] += right_val
    bucket_vals[left_bin] += left_val
    

def get_magnitude_hist_cell(loc_x, loc_y):
    # (loc_x, loc_y) defines the top left corner of the target cell.
    global G_x, G_y
    cell_x = G_x[loc_x:loc_x + CELL_SIZE, loc_y:loc_y + CELL_SIZE]
    cell_y = G_y[loc_x:loc_x + CELL_SIZE, loc_y:loc_y + CELL_SIZE]
    magnitudes = np.sqrt(cell_x * cell_x + cell_y * cell_y)
    directions = np.abs(np.arctan2(abs(cell_y), abs(cell_x)) * 180 / np.pi)
    
    rows, colls = cell_x.shape
    for i in range(rows):
        for j in range(colls):
            if cell_x[i][j] < 0 and cell_y[i][j] >= 0:
                directions[i][j] = 180 - directions[i][j]
            elif cell_x[i][j] < 0 and cell_y[i][j] < 0:
                directions[i][j] += 180
            elif cell_x[i][j] >= 0 and cell_y[i][j] < 0:
                directions[i][j] = 360 - directions[i][j]

    bucket_vals = np.zeros(N_BUCKETS)

    for i in range(len(directions.flatten())):
        assign_bucket_vals(magnitudes.flatten()[i], directions.flatten()[i], bucket_vals)
        
    return bucket_vals


def get_magnitude_hist_block(loc_x, loc_y):
    # (loc_x, loc_y) defines the top left corner of the target block.
    return reduce(
        lambda arr1, arr2: np.concatenate((arr1, arr2)),
        [get_magnitude_hist_cell(x, y) for x, y in zip(
            [loc_x, loc_x + CELL_SIZE, loc_x, loc_x + CELL_SIZE],
            [loc_y, loc_y, loc_y + CELL_SIZE, loc_y + CELL_SIZE],
        )]
    )

    
def get_image():
    # load image
    global G_x, G_y
    image = io.imread('figure.bmp')
    img =  rgb2gray(image)
    img = (img * 255).astype("uint8")
    
    # define filters to find edges
    kernel_x = np.array([[1,0,-1]])
    kernel_y = np.array([[-1],[0],[1]])
    
    # get edges
    G_x = sig.convolve2d(img, kernel_x, mode='same', boundary='symm') 
    G_y = sig.convolve2d(img, kernel_y, mode='same', boundary='symm') 
    
    # plot them
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 6), sharex=True, sharey=True)
    ax1.imshow(image); ax1.set_xlabel("Original")
    ax2.imshow(img, cmap=plt.cm.gray); ax2.set_xlabel("Gray scale")
    ax3.imshow((G_x + 255) / 2, cmap='gray'); ax3.set_xlabel("Gx")
    ax4.imshow((G_y + 255) / 2, cmap='gray'); ax4.set_xlabel("Gy")
    plt.show()


def get_histogram(loc_x, loc_y):
    get_image()
    
    ydata = get_magnitude_hist_block(loc_x, loc_y)
    ydata = ydata / np.linalg.norm(ydata)
    
    xdata = range(len(ydata))
    bucket_names = np.tile(np.arange(N_BUCKETS), BLOCK_SIZE * BLOCK_SIZE)
    
    assert len(ydata) == N_BUCKETS * (BLOCK_SIZE * BLOCK_SIZE)
    assert len(bucket_names) == len(ydata)
    
    # plot the histogram
    plt.figure(figsize=(15, 5))
    plt.bar(xdata, ydata, align='center', alpha=0.8, width=0.9)
    plt.xticks(xdata, bucket_names * 20, rotation=90)
    plt.xlabel('Direction buckets')
    plt.ylabel('Magnitude')
    plt.grid(ls='--', color='k', alpha=0.1)
    plt.title("HOG of block at [%d, %d]" % (loc_x, loc_y))
    plt.tight_layout()


loc_x = loc_y = 0
get_histogram(loc_x, loc_y)