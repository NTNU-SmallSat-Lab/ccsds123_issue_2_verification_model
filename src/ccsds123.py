from . import header as hd
from . import constants as const
from . import predictor as pred
from . import sa_encoder as sa_enc
import numpy as np
import time

class CCSDS123():
    """
    CCSDS 123.0-B-2 high level model class
    """
    header = None
    predictor = None
    raw_image_folder = "raw_images"
    image_sample = None # Symbol: s
    output_folder = "output"

    def __init__(self, image_name):
        self.image_name = image_name

    def __load_raw_image(self):
        """Load a raw image into a N_x * N_y by N_z array"""
        # Doing it the obvious way skipped the first byte, hence some hoops have been jumped through to fix it
        file = open(self.raw_image_folder + "/" + self.image_name, 'rb').read()
        first = file[0] # Set aside first byte
        last = (file[-2] << 8) + file[-1] # set aside last two bytes
        with open(self.raw_image_folder + "/" + self.image_name, 'rb') as file:
            file.seek(1) # Skip the first byte
            self.image_sample = np.fromfile(file, dtype=np.uint16)
        if self.image_sample.shape[0] != self.header.z_size * self.header.y_size * self.header.x_size:
            self.image_sample=np.pad(self.image_sample, (0,1), 'constant', constant_values=last) # Pad the last two bytes
        self.image_sample[0] += first << 8 # Reintroduce the first byte
        self.image_sample = self.image_sample.reshape((self.header.z_size, self.header.y_size, self.header.x_size)) # Reshape to z,y,x (BSQ) 3D array
        self.image_sample = self.image_sample.transpose(1,2,0) # Transpose to y,x,z order (BIP) 


    def compress_image(self):
        start_time = time.time()

        self.header = hd.Header(self.image_name)
        self.header.set_encoding_order_bip()

        self.__load_raw_image()
        print(f"{time.time() - start_time:.3f} seconds. Done with loading")

        self.image_constants = const.ImageConstants(self.header)

        self.predictor = pred.Predictor(self.header, self.image_constants, self.image_sample)
        self.predictor.run_predictor()
        print(f"{time.time() - start_time:.3f} seconds. Done with predictor")

        self.encoder = sa_enc.SampleAdaptiveEncoder(self.header,
                                                    self.image_constants,
                                                    self.predictor.get_predictor_output())
        self.encoder.run_encoder()
        print(f"{time.time() - start_time:.3f} seconds. Done with encoder")

        self.predictor.save_data(self.output_folder)
        self.encoder.save_data(self.output_folder)
        print(f"{time.time() - start_time:.3f} seconds. Done with saving")
        



