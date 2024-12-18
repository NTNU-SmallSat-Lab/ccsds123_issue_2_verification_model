from . import header as hd
from .hybrid_encoder_tables import *
import numpy as np
from bitarray import bitarray
from math import ceil, log2

class HybridEncoder():

    header = None
    image_constants = None
    mapped_quantizer_index = None # Symbol: delta
    accu_init_file = None
    use_accu_init_file = False
    
    def __init__(self, header, image_constants, mapped_quantizer_index):
        self.header = header
        self.image_constants = image_constants
        self.mapped_quantizer_index = mapped_quantizer_index

    unary_length_limit = None # Symbol: U_max
    rescaling_counter_size = None # Symbol: gamma*
    initial_count_exponent = None # Symbol: gamma_0

    def __init_encoder_constants(self):
        self.unary_length_limit = self.header.unary_length_limit + 32 * (self.header.unary_length_limit == 0)
        self.rescaling_counter_size = self.header.rescaling_counter_size + 4
        self.initial_count_exponent = self.header.initial_count_exponent + 8 * (self.header.initial_count_exponent == 0)
        tables_init()


    accumulator = None # Symbol: Sigma
    counter = None # Symbol: Gamma
    variable_length_code = None # Symbol: k
    bitstream = None
    bitstream_readable = None
    active_prefix = None
    code_index = None # Symbol: i
    input_symbol = None # Symbol: iota
    low_entropy_codes = None
    high_entropy_codes = None
    rescale_bits = None 
    flush_codes = None
    accumulator_final = None
    codewords = None
    codewords_binary = None
    entropy_type = None
    current_active_prefix = None
    prefix_match_index = None


    def __init_encoder_arrays(self):
        image_shape = self.mapped_quantizer_index.shape
        self.accumulator = np.zeros(image_shape, dtype=np.int64)
        self.counter = np.zeros(image_shape[:2], dtype=np.int64)
        self.variable_length_code = np.full(image_shape, fill_value=-1, dtype=np.int64)
        self.active_prefix = [''] * 16
        self.code_index = np.full(image_shape, fill_value=-1, dtype=np.int64)
        self.input_symbol = np.full(image_shape, fill_value='', dtype='U16')
        self.low_entropy_codes = np.full(image_shape, fill_value='', dtype='U85')
        self.high_entropy_codes = np.full(image_shape, fill_value='', dtype='U64')
        self.rescale_bits = np.full(image_shape, fill_value='', dtype='U1')
        self.flush_codes = np.full((16), fill_value='', dtype='U10')
        self.accumulator_final = np.full((self.header.z_size), fill_value='', dtype='U46')
        self.codewords = np.full(image_shape, fill_value='', dtype='U16')
        self.codewords_binary = np.full(image_shape, fill_value='', dtype='U16')
        self.entropy_type = np.full(image_shape, fill_value=2, dtype=np.uint8)
        self.current_active_prefix = np.full(image_shape, fill_value='-', dtype='U16')
        self.prefix_match_index = np.full(image_shape, fill_value=-2, dtype=np.int64)

        self.counter[0,0] = 2**self.initial_count_exponent
        
        if self.use_accu_init_file:
            with open(self.accu_init_file, "rb") as file:
                data = file.read()
                data = ''.join([bin(byte)[2:].zfill(8) for byte in data])
                num_bits = self.image_constants.dynamic_range_bits + self.initial_count_exponent
                data = data[:num_bits * self.header.z_size] # Remove fill bits
                data = [int(data[i:i+num_bits], 2) for i in range(0, len(data), num_bits)]
                assert len(data) == self.header.z_size
                self.accumulator[0,0] = np.array(data)
        else:
            # The hybrid encoder does not have a default accumulator init value in the standard. This is an arbitrary value
            self.accumulator[0,0] = 4 * self.counter[0,0]

        self.bitstream = bitarray()
        self.bitstream_readable = np.full(image_shape, fill_value='', dtype='U86')

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

        if self.counter[prev_y,prev_x] == 2**self.rescaling_counter_size - 1:
            self.counter[y,x] = (self.counter[prev_y,prev_x] + 1) // 2
            self.accumulator[y,x,z] = \
                    (self.accumulator[prev_y,prev_x,z] + \
                    4 * self.mapped_quantizer_index[y,x,z] + 1) \
                    // 2
            accumulator_lsb = bin(self.accumulator[prev_y,prev_x,z])[-1]
            self.__add_to_bitstream(accumulator_lsb, x, y, z)
            self.rescale_bits[y,x,z] = accumulator_lsb
        else:
            self.counter[y,x] = self.counter[prev_y,prev_x] + 1
            self.accumulator[y,x,z] = \
                self.accumulator[prev_y,prev_x,z] + \
                4 * self.mapped_quantizer_index[y,x,z]
        
        if self.accumulator[y,x,z] * 2**14 >= threshold[0] * self.counter[y,x]:
            self.__encode_high_entropy(x, y, z)
            self.entropy_type[y,x,z] = 1
        else:
            self.__encode_low_entropy(x, y, z)
            self.entropy_type[y,x,z] = 0
        
    
    def __encode_high_entropy(self, x, y, z):
        self.variable_length_code[y,x,z] = min(
                log2((self.accumulator[y,x,z] + self.counter[y,x] * 49 // 2**5) // self.counter[y,x]) - 2,
                max(self.image_constants.dynamic_range_bits - 2, 2)
            )
        assert self.variable_length_code[y,x,z] >= 2

        code = self.__reverse_gpo2(self.mapped_quantizer_index[y,x,z], self.variable_length_code[y,x,z])
        self.__add_to_bitstream(code, x, y, z)
        self.high_entropy_codes[y,x,z] = code

    
    def __reverse_gpo2(self, j, k):
        bitstring_j = bin(j)[2:].zfill(self.image_constants.dynamic_range_bits)
        zeros = j // 2**k
        if zeros < self.unary_length_limit:
            if k == 0:
                return '1' + '0' * zeros
            return bitstring_j[-k:] + '1' + '0' * zeros
        else:
            return bitstring_j + '0' * self.unary_length_limit
        

    def __encode_low_entropy(self, x, y, z):
        code_index = -2
        for i in range(15, -1, -1):
            if self.accumulator[y,x,z] * 2**14 < self.counter[y,x] * threshold[i]:
                code_index = i
                break
        self.code_index[y,x,z] = code_index
        input_symbol = "F"
        if self.mapped_quantizer_index[y,x,z] <= input_symbol_limit[code_index]:
            input_symbol = hex(self.mapped_quantizer_index[y,x,z])[2:].upper()
            assert 'h' not in self.input_symbol[y,x,z]
        else:
            input_symbol = 'X'
            residual_value = self.mapped_quantizer_index[y,x,z] - input_symbol_limit[code_index] - 1
            code = self.__reverse_gpo2(residual_value, 0)
            self.__add_to_bitstream(code, x, y, z)
            self.low_entropy_codes[y,x,z] += code
        
        self.active_prefix[code_index] += input_symbol
        self.current_active_prefix[y,x,z] = self.active_prefix[code_index]
        self.input_symbol[y,x,z] = input_symbol
        prefix_match_index = np.where(code_table_input[code_index] == self.active_prefix[code_index])[0]
        self.prefix_match_index[y,x,z] = prefix_match_index[0] if prefix_match_index.shape[0] == 1 else -1

        if prefix_match_index.shape[0] == 1:
            codeword = code_table_output[code_index][prefix_match_index[0]]
            self.codewords[y,x,z] = codeword
            assert 'Z' not in codeword
            codeword_binary = self.__table_codeword_to_binary(codeword)
            self.codewords_binary[y,x,z] = codeword_binary
            self.__add_to_bitstream(codeword_binary, x, y, z)
            self.low_entropy_codes[y,x,z] += codeword_binary

            self.active_prefix[code_index] = ''


    def __table_codeword_to_binary(self, codeword):
        return bin(int(codeword.split("'h")[1], 16))[2:].zfill(int(codeword.split("'h")[0]))


    def __add_to_bitstream(self, bitstring, x=None, y=None, z=None):
        self.bitstream += bitstring
        if x is not None and y is not None and z is not None:
            self.bitstream_readable[y,x,z] += bitstring


    def __encode_error_limits(self, y):
        period_index = y // 2**self.header.error_update_period_exponent
        if self.header.quantizer_fidelity_control_method != hd.QuantizerFidelityControlMethod.RELATIVE_ONLY:
            if self.header.absolute_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                self.__add_to_bitstream(
                    bin(self.header.periodic_absolute_error_limit_table[period_index][0])[2:].zfill(self.header.get_absolute_error_limit_bit_depth_value()),
                    0, y, 0
                )
            elif self.header.absolute_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                for z in range(self.header.z_size):
                    self.__add_to_bitstream(
                        bin(self.header.periodic_absolute_error_limit_table[period_index][z])[2:].zfill(self.header.get_absolute_error_limit_bit_depth_value()),
                        0, y, z
                    )

        if self.header.quantizer_fidelity_control_method != hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
            if self.header.relative_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                self.__add_to_bitstream(
                    bin(self.header.periodic_relative_error_limit_table[period_index][0])[2:].zfill(self.header.get_relative_error_limit_bit_depth_value()),
                    0, y, 0
                )
            elif self.header.relative_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                for z in range(self.header.z_size):
                    self.__add_to_bitstream(
                        bin(self.header.periodic_relative_error_limit_table[period_index][z])[2:].zfill(self.header.get_relative_error_limit_bit_depth_value()),
                        0, y, z
                    )

    
    def __encode_image_tail(self):
        for i in range(16):
            index = np.where(flush_table_prefix[i] == self.active_prefix[i])[0][0] if self.active_prefix[i] != '' else 0
            code = self.__table_codeword_to_binary(flush_table_word[i][index])
            self.__add_to_bitstream(code)
            self.flush_codes[i] = code
        
        for z in range(self.header.z_size):
            code = bin(self.accumulator[self.header.y_size - 1, self.header.x_size - 1,z])[2:]
            code = code.zfill(2 + self.image_constants.dynamic_range_bits + self.rescaling_counter_size)
            self.__add_to_bitstream(code)
            self.accumulator_final[z] = code

        self.__add_to_bitstream('1')
    

    def set_hybrid_accu_init_file(self, accu_init_file):
        self.accu_init_file = accu_init_file
        self.use_accu_init_file = True    


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
        
        self.__encode_image_tail()
        print("")
        

    def save_data(self, output_folder, header_bitstream):
        full_bitstream = header_bitstream + self.bitstream
        
        # Pad to word size
        word_bits = 8 * (self.header.output_word_size + 8 * (self.header.output_word_size == 0))
        fill_bits = (word_bits - (len(full_bitstream)) % word_bits) % word_bits
        full_bitstream += '0' * fill_bits

        with open(output_folder + "/z-output-bitstream.bin", "wb") as file:
            full_bitstream.tofile(file)
      
        # Save the initial accumulator value
        accu = bitarray()
        for z in range(self.header.z_size):
            accu += bin(int(self.accumulator[0,0,z]))[2:].zfill(int(self.image_constants.dynamic_range_bits + self.initial_count_exponent))
        accu += '0' * (8 - len(accu))
        with open(output_folder + "/hybrid_initial_accumulator.bin", "wb") as file:
            accu.tofile(file)

        csv_image_shape = (self.header.y_size * self.header.x_size, self.header.z_size)
        np.savetxt(output_folder + "/hybrid-encoder-00-accumulator.csv", self.accumulator.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/hybrid-encoder-01-counter.csv", self.counter.reshape(csv_image_shape[:1]), delimiter=",", fmt='%d') 
        np.savetxt(output_folder + "/hybrid-encoder-02-variable-length-code.csv", self.variable_length_code.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/hybrid-encoder-03-code-index.csv", self.code_index.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/hybrid-encoder-04-input-symbol.csv", self.input_symbol.reshape(csv_image_shape), delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/hybrid-encoder-05-current-active-prefix.csv", self.current_active_prefix.reshape(csv_image_shape), delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/hybrid-encoder-06-codewords.csv", self.codewords.reshape(csv_image_shape), delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/hybrid-encoder-07-codewords-binary.csv", self.codewords_binary.reshape(csv_image_shape), delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/hybrid-encoder-08-entropy-type.csv", self.entropy_type.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/hybrid-encoder-09-bitstream-readable.csv", self.bitstream_readable.reshape(csv_image_shape), delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/hybrid-encoder-10-prefix-match-index.csv", self.prefix_match_index.reshape(csv_image_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/hybrid-encoder-11-low_entropy_codes.csv", self.low_entropy_codes.reshape(csv_image_shape), delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/hybrid-encoder-12-high_entropy_codes.csv", self.high_entropy_codes.reshape(csv_image_shape), delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/hybrid-encoder-13-rescale_bits.csv", self.rescale_bits.reshape(csv_image_shape), delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/hybrid-encoder-14-flush_codes.csv", self.flush_codes, delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/hybrid-encoder-15-accumulator_final.csv", self.accumulator_final, delimiter=",", fmt='%s')
