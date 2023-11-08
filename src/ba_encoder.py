from . import header as hd
from . import constants as const
import numpy as np
from bitarray import bitarray
from math import ceil, log2


class BlockAdaptiveEncoder():

    header = None
    image_constants = None
    mapped_quantizer_index = None # Symbol: delta

    def __init__(self, header, image_constants, mapped_quantizer_index):
        self.header = header
        self.image_constants = image_constants
        self.mapped_quantizer_index = mapped_quantizer_index

    # unary_length_limit = None # Symbol: U_max
    # accumulator_init_parameter_1 = None # Symbol: k'
    # accumulator_init_parameter_2 = None # Symbol: k''
    # rescaling_counter_size = None # Symbol: gamma*
    # initial_count_exponent = None # Symbol: gamma_0
    
    def __init_encoder_constants(self):        
        # self.unary_length_limit = self.header.unary_length_limit + 32 * (self.header.unary_length_limit == 0)
        # self.rescaling_counter_size = self.header.rescaling_counter_size + 4
        # self.initial_count_exponent = self.header.initial_count_exponent + 8 * (self.header.initial_count_exponent == 0)

    # accumulator = None # Symbol: Sigma
    # counter = None # Symbol: Gamma
    # variable_length_code = None # Symbol: k
    # bitstream = None
    # bitstream_readable = None

    def __init_encoder_arrays(self):
        image_shape = self.mapped_quantizer_index.shape
        # self.accumulator = np.zeros(image_shape, dtype=np.int64)
        # self.counter = np.zeros(image_shape[:2], dtype=np.int64)
        # self.variable_length_code = np.zeros(image_shape, dtype=np.int64)

        self.bitstream = bitarray()
        self.bitstream_readable = np.zeros(image_shape, dtype='U64')

    def __encode_sample(self, x, y, z):
        exit()
    
    def __add_to_bitstream(self, bitstring, x, y, z):
        self.bitstream += bitstring
        self.bitstream_readable[y,x,z] = bitstring

    def run_encoder(self):
        self.__init_encoder_constants()
        self.__init_encoder_arrays()

        if self.header.sample_encoding_order == hd.SampleEncodingOrder.BI:
            for y in range(self.header.y_size):
                print(f"\rProcessing line y={y+1}/{self.header.y_size}", end="")

                if y % 2**self.header.error_update_period_exponent == 0 \
                    and self.header.periodic_error_updating_flag == \
                    hd.PeriodicErrorUpdatingFlag.USED:
                    exit("Periodic error updating flag not implemented")
                
                for i in range(ceil(self.header.z_size / self.header.sub_frame_interleaving_depth)):
                    for x in range(self.header.x_size):
                        z_start = i * self.header.sub_frame_interleaving_depth
                        z_end = min(
                            (i + 1) * (self.header.sub_frame_interleaving_depth),
                            self.header.z_size
                        )

                        for z in range(z_start, z_end):
                            self.__encode_sample(x, y, z)

        elif self.header.sample_encoding_order == hd.SampleEncodingOrder.BSQ:
            for z in range(self.header.z_size):
                print(f"\rProcessing band z={z+1}/{self.header.z_size}", end="")
                for y in range(self.header.y_size):
                    for x in range(self.header.x_size):
                        self.__encode_sample(x, y, z)
        print("")
    
    def save_data(self, output_folder, header_bitstream):
        self.bitstream = header_bitstream + self.bitstream
        
        # Pad to word size
        word_bits = 8 * (self.header.output_word_size + 8 * (self.header.output_word_size == 0))
        fill_bits = word_bits - (len(self.bitstream)) % word_bits
        self.bitstream += '0' * fill_bits

        with open(output_folder + "/z-output-bitstream.bin", "wb") as file:
            self.bitstream.tofile(file)

        csv_image_shape = (self.header.y_size * self.header.x_size, self.header.z_size)
        # np.savetxt(output_folder + "/sa-encoder-00-accumulator-init-parameter-1.csv", self.accumulator_init_parameter_1, delimiter=",", fmt='%d')
        # np.savetxt(output_folder + "/sa-encoder-01-accumulator-init-parameter-2.csv", self.accumulator_init_parameter_2, delimiter=",", fmt='%d')
        # np.savetxt(output_folder + "/sa-encoder-02-accumulator.csv", self.accumulator.reshape(csv_image_shape), delimiter=",", fmt='%d')
        # np.savetxt(output_folder + "/sa-encoder-03-counter.csv", self.counter.reshape(csv_image_shape[:1]), delimiter=",", fmt='%d') 
        # np.savetxt(output_folder + "/sa-encoder-04-bitstream-readable.csv", self.bitstream_readable.reshape(csv_image_shape), delimiter=",", fmt='%s')
        # np.savetxt(output_folder + "/sa-encoder-05-variable-length-code.csv", self.variable_length_code.reshape(csv_image_shape), delimiter=",", fmt='%d')
