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

    block_size = None # Symbol: J
    reference_sample_interval = None # Symbol: r
    id_bits = None # bits used for compression type identification. Zero-block and second extension have 1 more bit
    segment_size = 64 # Symbol: s. Number of blocks in a segment
    max_sample_split_bits = 0 # Maximum value for k
    
    def __init_encoder_constants(self):
        block_sizes = [8, 16, 32, 64]
        self.block_size = block_sizes[self.header.block_size]
        self.reference_sample_interval = self.header.reference_sample_interval + 2**12 * int(self.header.reference_sample_interval == 0)
        id_bits_lower_limit = 1 if self.header.restricted_code_options_flag == hd.RestrictedCodeOptionsFlag.RESTRICTED else 3
        self.id_bits = max(ceil(log2(self.image_constants.dynamic_range_bits)), id_bits_lower_limit)
        if self.header.restricted_code_options_flag == hd.RestrictedCodeOptionsFlag.RESTRICTED: 
            self.max_sample_split_bits = -1 if int(self.image_constants.dynamic_range_bits <= 2) else 1
        elif self.header.restricted_code_options_flag == hd.RestrictedCodeOptionsFlag.UNRESTRICTED:
            self.max_sample_split_bits = 5 if int(self.image_constants.dynamic_range_bits <= 8) else 13
            self.max_sample_split_bits = self.max_sample_split_bits if int(self.image_constants.dynamic_range_bits <= 16) else 29

    blocks = None
    blocks_shape = None
    encoding_results = None
    codes_binary = None
    zero_block_count = None
    bitstream = None
    bitstream_readable = None

    def __init_encoder_arrays(self):
        image_shape = self.mapped_quantizer_index.shape
        image_size = image_shape[0] * image_shape[1] * image_shape[2]
        self.blocks = np.zeros((image_size // self.block_size + int(image_size % self.block_size != 0), self.block_size), dtype=np.int64)
        self.blocks_shape = self.blocks.shape
        compressors_results_num = 1 + self.max_sample_split_bits + 1 + 1 # Second extension, sample splitting + 1 for k=0, no compression
        self.encoding_results = np.full((self.blocks.shape[0], compressors_results_num), fill_value='', dtype='U4096')
        self.zero_block_count = np.zeros((self.blocks.shape[0]), dtype=np.int64)

        self.bitstream = bitarray()
        self.bitstream_readable = np.full((self.blocks.shape[0]), fill_value='', dtype='U4096')

    def __encode_block(self, num):
        start_of_segment = (num % self.reference_sample_interval) % self.segment_size == 0
        zero_block = np.all(self.blocks[num] == 0)
        if zero_block:
            # This is a zero block
            if start_of_segment:
                self.zero_block_count[num] = 1
                if num == 0:
                    return
            else:
                self.zero_block_count[num] = self.zero_block_count[num - 1] + 1
                return
        
        # Check if previous was a zero block
        if self.zero_block_count[num - 1] > 0:
            code = '0' * (self.id_bits + 1)
            if self.zero_block_count[num - 1] <= 4:
                code += '0' * (self.zero_block_count[num - 1] - 1) + '1'
            elif start_of_segment:
                code += '00001'
            else:
                code += '0' * self.zero_block_count[num - 1] + '1'
                assert self.zero_block_count[num - 1] <= self.segment_size
            self.__add_to_bitstream(code, num - 1)
        if start_of_segment and zero_block:
            return

        self.encoding_results[num][0] = self.__encode_no_compression(num)
        self.encoding_results[num][1] = self.__encode_second_extension(num)
        for k in range(self.max_sample_split_bits + 1):
            self.encoding_results[num][k + 2] = self.__encode_sample_splitting(num, k)

        lowest_value = 4096
        lowest_index = 0
        for i in range(self.encoding_results.shape[1]):
            if len(self.encoding_results[num][i]) < lowest_value:
                lowest_value = len(self.encoding_results[num][i])
                lowest_index = i
        self.__add_to_bitstream(self.encoding_results[num][lowest_index], num)      

        
    def __encode_no_compression(self, num):
        return '1' * self.id_bits + ''.join([bin(self.blocks[num][i])[2:].zfill(self.image_constants.dynamic_range_bits) for i in range(self.block_size)])
    
    def __encode_second_extension(self, num):
        code = '0' * self.id_bits + '1'
        for i in range(0, self.block_size, 2):
            d0, d1 = self.blocks[num][i:i + 2]
            transformed = (d0 + d1) * (d0 + d1 + 1) // 2 + d1
            if transformed >= self.block_size * self.image_constants.dynamic_range_bits:
                # Abort. Outputs longer code than no compression
                return '0' * self.id_bits + '1' + '0' * (self.block_size + 1) * self.image_constants.dynamic_range_bits
            code += '0' * transformed + '1'
        return code
    
    def __encode_sample_splitting(self, num, k):
        fs_codes, split_codes = '', ''
        for i in range(self.block_size):
            fs_codes += '0' * int(bin(self.blocks[num][i])[2:].zfill(32)[:-k], 2) + '1' if k != 0 else '0' * self.blocks[num][i] + '1'
            split_codes += bin(self.blocks[num][i])[2:].zfill(33)[-k:] if k != 0 else ''
        return bin(k + 1)[2:].zfill(self.id_bits) + fs_codes + split_codes

    
    def __add_to_bitstream(self, bitstring, num):
        self.bitstream += bitstring
        self.bitstream_readable[num] = bitstring
    

    # def __encode_error_limits(self, y, num):
    #     period_index = y // 2**self.header.error_update_period_exponent
    #     if self.header.quantizer_fidelity_control_method != hd.QuantizerFidelityControlMethod.RELATIVE_ONLY:
    #         if self.header.absolute_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
    #             self.__add_to_bitstream(
    #                 bin(self.header.periodic_absolute_error_limit_table[period_index][0])[2:].zfill(self.header.get_absolute_error_limit_bit_depth_value()),
    #                 num
    #             )
    #         elif self.header.absolute_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_DEPENDENT:
    #             for z in range(self.header.z_size):
    #                 self.__add_to_bitstream(
    #                     bin(self.header.periodic_absolute_error_limit_table[period_index][z])[2:].zfill(self.header.get_absolute_error_limit_bit_depth_value()),
    #                     num
    #                 )

    #     if self.header.quantizer_fidelity_control_method != hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
    #         if self.header.relative_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
    #             self.__add_to_bitstream(
    #                 bin(self.header.periodic_relative_error_limit_table[period_index][0])[2:].zfill(self.header.get_relative_error_limit_bit_depth_value()),
    #                 num
    #             )
    #         elif self.header.relative_error_limit_assignment_method == hd.ErrorLimitAssignmentMethod.BAND_DEPENDENT:
    #             for z in range(self.header.z_size):
    #                 self.__add_to_bitstream(
    #                     bin(self.header.periodic_relative_error_limit_table[period_index][z])[2:].zfill(self.header.get_relative_error_limit_bit_depth_value()),
    #                     num
    #                 )


    def run_encoder(self):
        self.__init_encoder_constants()
        self.__init_encoder_arrays()

        if self.header.sample_encoding_order == hd.SampleEncodingOrder.BI: 
            self.blocks = self.blocks.reshape((self.blocks_shape[0] * self.blocks_shape[1]))
            index = 0

            # TODO: Do this cleaner and more efficiently
            for y in range(self.header.y_size):
                for i in range(ceil(self.header.z_size / self.header.sub_frame_interleaving_depth)):
                    for x in range(self.header.x_size):
                        z_start = i * self.header.sub_frame_interleaving_depth
                        z_end = min(
                            (i + 1) * (self.header.sub_frame_interleaving_depth),
                            self.header.z_size
                        )

                        for z in range(z_start, z_end):
                            self.blocks[index] = self.mapped_quantizer_index[y,x,z]
                            index += 1
            self.blocks = self.blocks.reshape((self.blocks_shape[0], self.blocks_shape[1]))

        elif self.header.sample_encoding_order == hd.SampleEncodingOrder.BSQ:
            self.blocks = self.mapped_quantizer_index.transpose(2,0,1) # Transpose to z,y,x order (BSQ)
            self.blocks = self.blocks.reshape((self.header.z_size * self.header.y_size * self.header.x_size)) # Reshape to 1D array
            padding = self.blocks_shape[0] * self.blocks_shape[1] - self.blocks.shape[0]
            self.blocks = np.pad(self.blocks, (0, padding), mode='constant', constant_values=0)
            self.blocks = self.blocks.reshape(self.blocks_shape)

        # prev_y = -1
        for num in range(self.blocks.shape[0]):
            print(f"\rProcessing block num={num+1}/{self.blocks.shape[0]}", end="")

                # The standard does not specify how the periodic error limits should be encoded. I added a solution based on the other coders, but test vector Test1-20190201/sample-000002 does not encode the periodic error limits at all, hence it is commented out
            # y = num * self.block_size // (self.header.x_size * self.header.z_size)
            # if self.header.periodic_error_updating_flag == \
            #     hd.PeriodicErrorUpdatingFlag.USED and \
            #     y != prev_y:
            #     prev_y = y
            #     if y % 2**self.header.error_update_period_exponent == 0:
            #         self.__encode_error_limits(y, num)
            self.__encode_block(num)
        
        if self.zero_block_count[-1] > 0:
            code = '0' * (self.id_bits + 1)
                # I can't see anywhere in the standards that specifies that the ROS codeword should be encoded if the last segment ends with a zero block at one less block than otherwise, but it seems to be the case from test vector Test1-20190201/sample-000005 and Test1-20190201/sample-000008
            if self.zero_block_count[-1] <= 3: 
                code += '0' * (self.zero_block_count[-1] - 1) + '1'
            else:
                code += '00001'
            self.__add_to_bitstream(code, self.blocks.shape[0] - 1)
        
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
        np.savetxt(output_folder + "/ba-encoder-00-blocks.csv", self.blocks.reshape(self.blocks_shape), delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/ba-encoder-01-encoding-results.csv", self.encoding_results, delimiter=",", fmt='%s')
        np.savetxt(output_folder + "/ba-encoder-02-zero-block-count.csv", self.zero_block_count, delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/ba-encoder-03-bitstream-readable.csv", self.bitstream_readable, delimiter=",", fmt='%s')

