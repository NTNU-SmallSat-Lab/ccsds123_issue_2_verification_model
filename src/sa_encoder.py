from . import header as hd
from . import constants as const
import numpy as np
from bitarray import bitarray
from math import ceil, log2


class SampleAdaptiveEncoder():

    header = None
    image_constants = None
    mapped_quantizer_index = None # Symbol: delta

    def __init__(self, header, image_constants, mapped_quantizer_index):
        self.header = header
        self.image_constants = image_constants
        self.mapped_quantizer_index = mapped_quantizer_index

    unary_length_limit = None # Symbol: U_max
    accumulator_init_parameter_1 = None # Symbol: k'
    accumulator_init_parameter_2 = None # Symbol: k''
    rescaling_counter_size = None # Symbol: gamma*
    initial_count_exponent = None # Symbol: gamma_0
    
    def __init_encoder_constants(self):        
        self.unary_length_limit = self.header.unary_length_limit + 32 * (self.header.unary_length_limit == 0)
        self.rescaling_counter_size = self.header.rescaling_counter_size + 4
        self.initial_count_exponent = self.header.initial_count_exponent + 8 * (self.header.initial_count_exponent == 0)

        if self.header.accumulator_init_constant != 15:
            self.accumulator_init_parameter_2 = np.full((self.header.z_size), self.header.accumulator_init_constant, dtype=np.int64)
        else:
            exit("Accumulator init table not implemented")        
        self.accumulator_init_parameter_1 = \
            (self.accumulator_init_parameter_2 <= 30 - self.image_constants.dynamic_range_bits).astype(int) * self.accumulator_init_parameter_2 + \
            (self.accumulator_init_parameter_2 > 30 - self.image_constants.dynamic_range_bits).astype(int) * (2 * self.accumulator_init_parameter_2 + self.image_constants.dynamic_range_bits - 30)


    accumulator = None # Symbol: Sigma
    counter = None # Symbol: Gamma
    variable_length_code = None # Symbol: k
    bitstream = None
    bitstream_readable = None

    def __init_encoder_arrays(self):
        image_shape = self.mapped_quantizer_index.shape
        self.accumulator = np.zeros(image_shape, dtype=np.int64)
        self.counter = np.zeros(image_shape[:2], dtype=np.int64)
        self.variable_length_code = np.zeros(image_shape, dtype=np.int64)

        if image_shape[1] > 1:
            self.counter[0,1] = 2**self.initial_count_exponent
            self.accumulator[0,1] = np.floor((3 * 2**(self.accumulator_init_parameter_1 + 6) - 49) * self.counter[0,1] // 2**7)

        self.bitstream = bitarray()
        self.bitstream_readable = np.full(image_shape, fill_value='', dtype='U512')

    def __encode_sample(self, x, y, z):
        if y == 0 and x == 0:
            bitstring = bin(self.mapped_quantizer_index[y,x,z])[2:].zfill(self.image_constants.dynamic_range_bits)
            self.__add_to_bitstream(bitstring,x,y,z)
            return
        
        prev_y = y
        prev_x = x - 1
        if prev_x < 0:
            prev_y -= 1
            prev_x = self.header.x_size - 1

        if y == 0 and x == 1:
            pass
        elif self.counter[prev_y,prev_x] == 2**self.rescaling_counter_size - 1:
            self.counter[y,x] = (self.counter[prev_y,prev_x] + 1) // 2
            self.accumulator[y,x,z] = \
                    (self.accumulator[prev_y,prev_x,z] + \
                    self.mapped_quantizer_index[prev_y,prev_x,z] + 1) \
                    // 2
        else:
            self.counter[y,x] = self.counter[prev_y,prev_x] + 1
            self.accumulator[y,x,z] = \
                self.accumulator[prev_y,prev_x,z] + \
                self.mapped_quantizer_index[prev_y,prev_x,z]
        
        self.__find_code_length(x,y,z)
        self.__add_to_bitstream(
            self.__gpo2(self.mapped_quantizer_index[y,x,z], self.variable_length_code[y,x,z]),
            x, y, z)
    
    def __find_code_length(self,x,y,z):
        if 2 * self.counter[y,x] > self.accumulator[y,x,z] + self.counter[y,x] * 49 // 2**7:
            self.variable_length_code[y,x,z] = 0
        else:
            self.variable_length_code[y,x,z] = \
                min(
                    (log2((self.accumulator[y,x,z] + self.counter[y,x] * 49 // 2**7) // self.counter[y,x])),
                    self.image_constants.dynamic_range_bits - 2
                )
        k_method_2 = 0
        for k in range(self.image_constants.dynamic_range_bits - 2, -1, -1):
            if self.counter[y,x] * 2**k <= self.accumulator[y,x,z] + self.counter[y,x] * 49 // 2**7:
                k_method_2 = k
                break
        assert k_method_2 == self.variable_length_code[y,x,z]            
                    
    def __gpo2(self, j, k):
        bitstring_j = bin(j)[2:].zfill(self.image_constants.dynamic_range_bits)
        zeros = j // 2**k
        if zeros < self.unary_length_limit:
            if k == 0:
                return '0' * zeros + '1'
            return '0' * zeros + '1' + bitstring_j[-k:]
        else:
            return '0' * self.unary_length_limit + bitstring_j
        
    def __encode_error_limits(self, y):
        period_index = y // 2**self.header.error_update_period_exponent
        if self.header.quantizer_fidelity_control_method != hd.QuantizerFidelityControlMethod.RELATIVE_ONLY:
            if self.header.absolute_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                self.__add_to_bitstream(
                    bin(self.header.periodic_absolute_error_limit_table[period_index][0])[2:].zfill(self.header.absolute_error_limit_bit_depth),
                    0, y, 0
                )
            elif self.header.absolute_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                for z in range(self.header.z_size):
                    self.__add_to_bitstream(
                        bin(self.header.periodic_absolute_error_limit_table[period_index][z])[2:].zfill(self.header.absolute_error_limit_bit_depth),
                        0, y, z
                    )

        if self.header.quantizer_fidelity_control_method != hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
            if self.header.relative_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                self.__add_to_bitstream(
                    bin(self.header.periodic_relative_error_limit_table[period_index][0])[2:].zfill(self.header.relative_error_limit_bit_depth),
                    0, y, 0
                )
            elif self.header.relative_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                for z in range(self.header.z_size):
                    self.__add_to_bitstream(
                        bin(self.header.periodic_relative_error_limit_table[period_index][z])[2:].zfill(self.header.relative_error_limit_bit_depth),
                        0, y, z
                    )

    def __add_to_bitstream(self, bitstring, x, y, z):
        self.bitstream += bitstring
        self.bitstream_readable[y,x,z] += bitstring

    def run_encoder(self):
        self.__init_encoder_constants()
        self.__init_encoder_arrays()

        if self.header.sample_encoding_order == hd.SampleEncodingOrder.BI:
            for y in range(self.header.y_size):
                print(f"\rProcessing line y={y+1}/{self.header.y_size}", end="")

                if y % 2**self.header.error_update_period_exponent == 0 \
                    and self.header.periodic_error_updating_flag == \
                    hd.PeriodicErrorUpdatingFlag.USED:
                    self.__encode_error_limits(y)
                
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
        fill_bits = (word_bits - (len(self.bitstream)) % word_bits) % word_bits
        self.bitstream += '0' * fill_bits

        with open(output_folder + "/z-output-bitstream.bin", "wb") as file:
            self.bitstream.tofile(file)
        with open(output_folder + "/hybrid_initial_accumulator.bin", "wb") as file:
            bitarray().tofile(file) # Create empty file. To simplify creating scripts compatible with all entropy coder types

        csv_image_shape = (self.header.y_size * self.header.x_size, self.header.z_size)
        np.savetxt(output_folder + "/sa-encoder-00-accumulator-init-parameter-1.csv", self.accumulator_init_parameter_1, delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/sa-encoder-01-accumulator-init-parameter-2.csv", self.accumulator_init_parameter_2, delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/sa-encoder-02-accumulator.csv", self.accumulator.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/sa-encoder-03-counter.csv", self.counter.reshape(csv_image_shape[:1]), delimiter=",", fmt='%d') 
        np.savetxt(output_folder + "/sa-encoder-04-bitstream-readable.csv", self.bitstream_readable.reshape(csv_image_shape), delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/sa-encoder-05-variable-length-code.csv", self.variable_length_code.reshape(csv_image_shape), delimiter=",", fmt='%d')
