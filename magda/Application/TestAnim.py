import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pydicom import dcmread
from functions import display_multiple_img, print_dicom_info

import numpy as np
import matplotlib.pyplot as plt


file_path = "..\\numer2\\exam.dcm"
ds = dcmread(file_path)
images = ds.pixel_array
print_dicom_info(ds)


fig, ax = plt.subplots()
a = images[0]
im = ax.imshow(a, cmap='gray', animated=True)
time_text = ax.text(0.5, 0.05, "", bbox={'facecolor': 'red',
                                       'alpha': 0.5, 'pad': 2},
         transform=ax.transAxes, ha="center")

#
# def init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     return ln,

def init():
    im.set_data(np.random.random((5,5)))
    return [im]

# animation function.  This is called sequentially
def update(i):
    im.set_array(images[i])
    time_text.set_text(str(i))
    return im, time_text,

# def update(frame):
#     im.set_array(a)
#     im = ax.imshow(images[frame], cmap='gray', animated=True)
#     ax.set_title(frame)
#     # xdata.append(frame)
#     # ydata.append(np.sin(frame))
#     # ln.set_data(xdata, ydata)
#     return im,

ani = animation.FuncAnimation(fig, update, frames=22, blit=True, interval=100)
plt.show()


# DZIAŁA !!!
#
# file_path = "..\\numer2\\exam.dcm"
# ds = dcmread(file_path)
# images = ds.pixel_array
# print_dicom_info(ds)
#
#
# fig = plt.figure()
#
#
# # ims is a list of lists, each row is a list of artists to draw in the
# # current frame; here we are just animating one artist, the image, in
# # each frame
# ims = []
# for i in range(22):
#     im = plt.imshow(images[i], cmap='gray', animated=True)
#     ims.append([im])
#
#
# ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat=False)
#
# # ani.save('dynamic_images.mp4')
#
# plt.show()
