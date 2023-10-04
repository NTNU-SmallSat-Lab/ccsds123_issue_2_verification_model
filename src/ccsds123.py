from . import header as hd
import numpy as np


class CCSDS123():

    raw_image_folder = "raw_images"
    image_raw = None

    def __init__(self, image_name):
        self.image_name = image_name
        self.header = hd.Header(image_name)

    # Load a raw image into a N_x * N_y by N_z array
    def load_raw_image(self):
        # Doing it the obvious way skipped the first byte, hence some hoops have been jumped through to fix it
        file = open(self.raw_image_folder + "/" + self.image_name, 'rb').read()
        first = file[0]
        last = (file[-2] << 8) + file[-1]
        with open(self.raw_image_folder + "/" + self.image_name, 'rb') as f:
            f.seek(1)
            self.image_raw = np.fromfile(f, dtype=np.uint16)
        self.image_raw=np.pad(self.image_raw, (0,1), 'constant', constant_values=last)
        self.image_raw[0] += first << 8
        self.image_raw.shape = (self.header.z_size, self.image_raw.size//self.header.z_size)
