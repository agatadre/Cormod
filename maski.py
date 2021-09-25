from pydicom import dcmread
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage import img_as_ubyte
import math

def display_multiple_img(images, rows=1, type='gray', title='', addColorbar=False):
    cols = math.ceil(images.shape[0]/rows)
    fig, axs = plt.subplots( nrows=rows, ncols=cols )
    num_of_img = images.shape[0]
    axs = axs.ravel()
    for ind in range(num_of_img):
        img_ret = axs[ind].imshow( images[ind], type, aspect='equal' )
        axs[ind].set_title( ind, fontsize=8, pad=0.1 )
        axs[ind].set_axis_off()
    for ind in range(num_of_img, rows*cols):
        axs[ind].set_axis_off()
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.2)
    if title != '':
        fig.canvas.set_window_title(title)
    if addColorbar:
        fig.colorbar(img_ret)

#obrazek (y,x)
def clearImageEdges(img):
    imgM=np.copy(img)
    imgM[0:512, 0:30]=0
    imgM[0:43, 0:512]=0
    imgM[0:512,475:512]=0
    imgM[470:512, 0:512]=0
    return imgM

def maskImage(img, threshold=100):
    imgM = np.copy(img)
    kernel = np.ones((5, 5), np.uint8)
    imgM = cv2.dilate(img, kernel, iterations=3)
    #imgM = cv2.bilateralFilter(imgM, 100, 20, 600, borderType=cv2.BORDER_CONSTANT)
    for i in range(imgM.shape[0]):
        for j in range(imgM.shape[1]):
            if imgM[i][j] < threshold:
                imgM[i][j] = 0
            else:
                imgM[i][j] = 1

    return imgM


ds = dcmread("exam.dcm")


ind=21
img_2d = ds.pixel_array[ind]
img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0

frangi_img = frangi(img_2d_scaled)
img0 = img_as_ubyte(frangi_img)
print(img0)


img1=clearImageEdges(img0)
#img1=cv2.equalizeHist(img1)
img1=maskImage(img1, 90)

print(img1)
plt.subplot(1, 2, 1)
plt.imshow(img0, "gray")
plt.subplot(1, 2, 2)
plt.imshow(img1, "gray")

#display_multiple_img(ds.pixel_array, 7)
plt.show()
