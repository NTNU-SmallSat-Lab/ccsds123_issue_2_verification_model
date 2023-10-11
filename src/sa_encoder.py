from . import header as hd
from . import constants as const
import numpy as np
from math import ceil


class SampleAdaptiveEncoder():

    header = None
    image_constants = None
    mapped_quantizer_index = None # Symbol: delta

    def __init__(self, header, image_constants, mapped_quantizer_index):
        self.header = header
        self.image_constants = image_constants
        self.mapped_quantizer_index = mapped_quantizer_index

    accumulator_init_parameter_1 = None # Symbol: k'
    accumulator_init_parameter_2 = None # Symbol: k''
    
    def __init_encoder_constants(self):        
        if self.header.accumulator_init_constant != 0:
            self.accumulator_init_parameter_2 = np.full((self.header.z_size), self.header.accumulator_init_constant, dtype=np.int64)
        else:
            exit("Accumulator init table not implemented")        
        self.accumulator_init_parameter_1 = \
            (self.accumulator_init_parameter_2 <= 30 - self.image_constants.dynamic_range_bits).astype(int) * self.accumulator_init_parameter_2 + \
            (self.accumulator_init_parameter_2 > 30 - self.image_constants.dynamic_range_bits).astype(int) * (2 * self.accumulator_init_parameter_2 + self.image_constants.dynamic_range_bits - 30)

    

    accumulator = None # Symbol: Sigma
    counter = None # Symbol: Gamma

    def __init_encoder_arrays(self):
        image_shape = self.mapped_quantizer_index.shape
        self.accumulator = np.zeros(image_shape, dtype=np.int64)
        self.counter = np.zeros(image_shape[:2], dtype=np.int64)

        self.counter[0,1] = 2**self.header.initial_count_exponent
        self.accumulator[0,1] = np.floor((3 * 2**(self.accumulator_init_parameter_1 + 6) - 49) * self.counter[0,1] / 2**7)
        

            



    def __encode_sample(self, x, y, z):
        return


    def run_encoder(self):
        self.__init_encoder_constants()
        self.__init_encoder_arrays()

        if self.header.sample_encoding_order == hd.SampleEncodingOrder.BI:
            for y in range(self.header.y_size):

                if y % 2**self.header.error_update_period_exponent == 0 \
                    and self.header.periodic_error_updating_flag == \
                    hd.PeriodicErrorUpdatingFlag.USED:
                    exit("Periodic error updating flag not implemented")
                
                for i in range(ceil(self.header.z_size / self.header.sub_frame_interleaving_depth)):
                    for x in range(self.header.x_size):
                        z_start = i * self.header.sub_frame_interleaving_depth
                        z_end = min(
                            (i + 1) * (self.header.sub_frame_interleaving_depth - 1),
                            self.header.z_size - 1
                        )

                        for z in range(z_start, z_end):
                            self.__encode_sample(x, y, z)

        elif self.header.sample_encoding_order == hd.SampleEncodingOrder.BSQ:
            exit("BSQ encoding order not implemented")
    
    def save_data(self, output_folder):
        csv_image_shape = (self.header.y_size * self.header.x_size, self.header.z_size)
        np.savetxt(output_folder + "/sa-encoder-00-accumulator-init-parameter-1.csv", self.accumulator_init_parameter_1, delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/sa-encoder-01-accumulator-init-parameter-2.csv", self.accumulator_init_parameter_2, delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/sa-encoder-02-accumulator.csv", self.accumulator.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/sa-encoder-03-counter.csv", self.counter.reshape(csv_image_shape[:1]), delimiter=",", fmt='%d') 

        
