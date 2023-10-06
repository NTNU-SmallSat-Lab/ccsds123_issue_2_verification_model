from . import header as hd
import numpy as np


class CCSDS123():

    header = None
    raw_image_folder = "raw_images"
    image_raw = None

    def __init__(self, image_name):
        self.image_name = image_name
        self.header = hd.Header(image_name)

    # Load a raw image into a N_x * N_y by N_z array
    def load_raw_image(self):
        # Doing it the obvious way skipped the first byte, hence some hoops have been jumped through to fix it
        file = open(self.raw_image_folder + "/" + self.image_name, 'rb').read()
        first = file[0] # Set aside first byte
        last = (file[-2] << 8) + file[-1] # set aside last two bytes
        with open(self.raw_image_folder + "/" + self.image_name, 'rb') as file:
            file.seek(1) # Skip the first byte
            self.image_raw = np.fromfile(file, dtype=np.uint16)
        self.image_raw=np.pad(self.image_raw, (0,1), 'constant', constant_values=last) # Pad the last two bytes
        self.image_raw[0] += first << 8 # Reintroduce the first byte
        self.image_raw.shape = (self.header.z_size, self.image_raw.size//self.header.z_size)

    local_sum = None
    local_difference_vector = None
    weight_vector = None
    predicted_central_local_difference = None
    high_resolution_predicted_sample_value = None
    double_resolution_predicted_sample_value = None
    predicted_sample_value = None
    prediction_residual = None
    maximum_error = None
    quantizer_index = None
    double_resolution_sample_representative = None
    sample_representative = None
    clipped_quantizer_bin_center = None
    double_resolution_prediction_error = None
    mapped_quantizer_index = None
    
    def create_predictor_arrays(self):
        image_shape = self.image_raw.shape
        self.local_difference_vector = np.empty(image_shape, dtype=np.int32)

        num_local_difference_values = self.header.prediction_bands_num
        if self.header.prediction_mode == hd.PredictionMode.FULL:
            num_local_difference_values += 3
        local_difference_vector_shape = (num_local_difference_values,) + image_shape
        self.local_difference_vector = np.empty(local_difference_vector_shape, dtype=np.int32)
        self.weight_vector = np.empty(local_difference_vector_shape, dtype=np.int32)
        self.predicted_central_local_difference = np.empty(image_shape, dtype=np.int32)
        self.high_resolution_predicted_sample_value = np.empty(image_shape, dtype=np.int32)
        self.double_resolution_predicted_sample_value = np.empty(image_shape, dtype=np.int32)
        self.predicted_sample_value = np.empty(image_shape, dtype=np.int32)
        self.prediction_residual = np.empty(image_shape, dtype=np.int32)
        self.maximum_error = np.empty(image_shape, dtype=np.int32)
        self.quantizer_index = np.empty(image_shape, dtype=np.int32)
        self.double_resolution_sample_representative = np.empty(image_shape, dtype=np.int32)
        self.sample_representative = np.empty(image_shape, dtype=np.int32)
        self.clipped_quantizer_bin_center = np.empty(image_shape, dtype=np.int32)
        self.double_resolution_prediction_error = np.empty(image_shape, dtype=np.int32)
        self.mapped_quantizer_index = np.empty(image_shape, dtype=np.int32)


    # def predictor(self):

