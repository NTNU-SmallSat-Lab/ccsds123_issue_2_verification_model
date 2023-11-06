from enum import Enum
import re
from bitarray import bitarray

class SampleType(Enum):
        UNSIGNED_INTEGER = 0
        SIGNED_INTEGER = 1

class LargeDFlag(Enum):
    SMALL_D = 0 # <=16 bit
    LARGE_D = 1 # >16 bit

class SampleEncodingOrder(Enum):
    BI  = 0
    BSQ = 1

class EntropyCoderType(Enum):
    SAMPLE_ADAPTIVE = 0
    HYBRID = 1
    BLOCK_ADAPTIVE = 2

class QuantizerFidelityControlMethod(Enum):
    LOSSLESS = 0
    ABSOLUTE_ONLY = 1
    RELATIVE_ONLY = 2
    ABSOLUTE_AND_RELATIVE = 3

class SampleRepresentativeFlag(Enum):
    NOT_INCLUDED = 0 # phi = psi = 0 for all bands
    INCLUDED = 1

class PredictionMode(Enum):
    FULL = 0
    REDUCED = 1

class WeightExponentOffsetFlag(Enum):
    ALL_ZERO = 0
    NOT_ALL_ZERO = 1

class LocalSumType(Enum):
    WIDE_NEIGHBOR_ORIENTED = 0
    NARROW_NEIGHBOR_ORIENTED = 1
    WIDE_COLUMN_ORIENTED = 2
    NARROW_COLUMN_ORIENTED = 3

class WeightExponentOffsetTableFlag(Enum):
    NOT_INCLUDED = 0
    INCLUDED = 1

class WeightInitMethod(Enum):
    DEFAULT = 0
    CUSTOM = 1

class WeightInitTableFlag(Enum):
    NOT_INCLUDED = 0
    INCLUDED = 1

class PeriodicErrorUpdatingFlag(Enum):
    NOT_USED = 0
    USED = 1

class ErrorLimitAssignmentMethod(Enum):
    BAND_INDEPENDENT = 0
    BAND_DEPENDENT = 1

class BandVaryingDampingFlag(Enum):
    BAND_INDEPENDENT = 0
    BAND_DEPENDENT = 1

class DampingTableFlag(Enum):
    NOT_INCLUDED = 0
    INCLUDED = 1

class BandVaryingOffsetFlag(Enum):
    BAND_INDEPENDENT = 0
    BAND_DEPENDENT = 1

class OffsetTableFlag(Enum):
    NOT_INCLUDED = 0
    INCLUDED = 1

class AccumulatorInitTableFlag(Enum):
    NOT_INCLUDED = 0
    INCLUDED = 1

class RestrictedCodeOptionsFlag(Enum):
    UNRESTRICTED = 0
    RESTRICTED = 1


class Header:
    """
    Header class for storing image metadata and configuration options.
    """

    ################
    # Image metadata
    ################
    user_defined_data = 0
    x_size = 0 # N_x. Encode as N_x%2^16. 1<=N_x<=2^16-1
    y_size = 0 # N_y. Encode as N_y%2^16. 1<=N_y<=2^16-1
    z_size = 0 # N_z. Encode as N_z%2^16. 1<=N_z<=2^16-1
    sample_type = SampleType.UNSIGNED_INTEGER
    large_d_flag = LargeDFlag.SMALL_D
    dynamic_range = 14 # D. Encode as D%16. 1<=D<=32
    sample_encoding_order = SampleEncodingOrder.BI
    sub_frame_interleaving_depth = 1 # M. Encode as M%2^16. M=1 for BIL, M=z_size for BIP. 1<=M<=z_size
    output_word_size = 1 # B. Encode as B%8. 1<=B<=8
    entropy_coder_type = EntropyCoderType.SAMPLE_ADAPTIVE
    quantizer_fidelity_control_method = QuantizerFidelityControlMethod.ABSOLUTE_AND_RELATIVE
    supplementary_information_table_count = 0 # tau. 0<=tau<=15. Supplementary information tables are not implemented

    # TODO: Support supplementary information tables

    #####################
    # Predicator metadata
    #####################
    sample_representative_flag = SampleRepresentativeFlag.INCLUDED
    prediction_bands_num = 4 # P. 0<=P<=15
    prediction_mode = PredictionMode.FULL
    weight_exponent_offset_flag = WeightExponentOffsetFlag.ALL_ZERO
    local_sum_type = LocalSumType.WIDE_NEIGHBOR_ORIENTED
    register_size = 32 # R. Encode as R%64. max{32,D+Omega+2}<=R<=64
    weight_component_resolution = 2 # Omega. Encode as Omega-4. 4<=Omega<=19
    weight_update_change_interval = 6 # t_inc. Encode as log2(t_inc)-4. 2^4<=t_inc<=2^11
    weight_update_initial_parameter = 6 # nu_min. Encode as nu_min+6. -6<=nu_min<=nu_max<=9
    weight_update_final_parameter = 10 # nu_max. Encode as nu_max+6. -6<=nu_min<=nu_max<=9
    weight_exponent_offset_table_flag = WeightExponentOffsetTableFlag.NOT_INCLUDED
    weight_init_method = WeightInitMethod.DEFAULT
    weight_init_table_flag = WeightInitTableFlag.NOT_INCLUDED
    weight_init_resolution = 0 # Q. Encode as 0 if weight_init_method=DEFAULT, otherwise as Q. 3<=Q<=Omega+3
    
    # TODO: Weight initialization table

    # TODO: Weight exponent offset table

    # Quantization
    # Error limit update
    periodic_error_updating_flag = PeriodicErrorUpdatingFlag.NOT_USED
    error_update_period_exponent = 0 # u. Encode as 0 if periodic_error_updating_flag=NOT_USED, otherwise as u. 0<=u<=9
    # Absolute error limit
    absolute_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_INDEPENDENT
    absolute_error_limit_bit_depth = 2 # D_A. Encode as D_A%16. 1<=D_A<=min{D − 1,16}
    absolute_error_limit_value = 2 # A*. 0<=A*<=2^D_A-1. TODO: Support BAND_DEPENDENT values
    # Relative error limit
    relative_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_INDEPENDENT
    relative_error_limit_bit_depth = 4 # D_R. Encode as D_R%16. 1<=D_R<=min{D − 1,16}
    relative_error_limit_value = 8 # R*. 0<=R*<=2^D_R-1. TODO: Support BAND_DEPENDENT values
    # TODO: Support periodic error updating

    # Sample Representative
    sample_representative_resolution = 4 # Theta. 0<=Theta<=4
    band_varying_damping_flag = BandVaryingDampingFlag.BAND_INDEPENDENT
    damping_table_flag = DampingTableFlag.NOT_INCLUDED
    fixed_damping_value = 0 # phi. Encode as 0 if damping_table_flag=INCLUDED, otherwise as phi. 0<=phi<=2^Theta-1
    band_varying_offset_flag = BandVaryingOffsetFlag.BAND_INDEPENDENT
    damping_offset_table_flag = OffsetTableFlag.NOT_INCLUDED
    fixed_offset_value = 0 # psi. Encode as 0 if damping_offset_table_flag=INCLUDED, otherwise as psi. 0<=psi<=2^Theta-1. psi=0 if lossless
    # TODO: Support damping table subblock
    # TODO: Support offset table subblock

    ########################
    # Entropy coder metadata
    ########################
    # Sample-adaptive entropy coder and Hybrid entropy coder
    unary_length_limit = 16 # U_max. Encode as U_max%32. 8<=U_max<=32
    rescaling_counter_size = 5 # gamma*. Encode as gamma*-4. Max{4,gamma_0+1}<=γ<=11
    initial_count_exponent = 5 # gamma_0. Encode as gamma_0%8. 1<=gamma_0<=8
    # Remaining sample-adaptive entropy coder
    accumulator_init_constant = 4 # K. Encode as 15 if K is not used. 0<=K<=min(D-2,14)
    accumulator_init_table_flag = AccumulatorInitTableFlag.NOT_INCLUDED
    # TODO: Support accumulator initialization table

    # Block-adaptive entropy coder
    block_size = 2 # B. 0: J=8, 1: J=16, 2: J=32, 3: J=64
    restricted_code_options_flag = RestrictedCodeOptionsFlag.UNRESTRICTED
    reference_sample_interval = 2**11 # r. Encode as r%2**12.

    header_bitstream = None

    def __init__(self, image_name):
        self.__set_config_according_to_image_name(image_name)
        self.__check_legal_config()
        
    def __set_config_according_to_image_name(self, image_name):
        # TODO: Actually learn regex and do this properly
        self.x_size = int(re.findall('x(.*).raw', image_name)[0].split("x")[-1]) 
        self.y_size = int(re.findall('x(.+)x', image_name)[0])
        self.z_size = int(image_name.split('x')[0].split('-')[-1])
        format = re.findall('-(.*)-', image_name)[0]
        format = image_name.split('-')[-2].split('-')[-1]
        self.sample_type = SampleType.UNSIGNED_INTEGER if format[0] == 'u' else SampleType.SIGNED_INTEGER

        # Pick dynamic range manually, since the range in the file name is generally way to wide
        # self.large_d_flag = LargeDFlag.SMALL_D if int(format[1:3]) <= 16 else LargeDFlag.LARGE_D
        # self.dynamic_range = int(format[1:3]) % 16

        if format[3:5] != 'be':
            exit("Only big endian is supported")
    
    def set_config_from_file(self, config_file):
        bitstream = bitarray()
        # print(f"Reading configuration from {config_file}")
        with open(config_file, "rb") as file:
            bitstream.fromfile(file)

            # Image metadata 
        # Essential subpart
        self.user_defined_data = int(bitstream[0:8].to01(), 2)
        self.x_size = int(bitstream[8:24].to01(), 2)
        self.y_size = int(bitstream[24:40].to01(), 2)
        self.z_size = int(bitstream[40:56].to01(), 2)
        self.sample_type = SampleType(int(bitstream[56:57].to01(), 2))
        assert bitstream[57:58].to01() == '0' # Reserved
        self.large_d_flag = LargeDFlag(int(bitstream[58:59].to01(), 2))
        self.dynamic_range = int(bitstream[59:63].to01(), 2)
        self.sample_encoding_order = SampleEncodingOrder(int(bitstream[63:64].to01(), 2))
        self.sub_frame_interleaving_depth = int(bitstream[64:80].to01(), 2)
        assert bitstream[80:82].to01() == '00' # Reserved
        self.output_word_size = int(bitstream[82:85].to01(), 2)
        self.entropy_coder_type = EntropyCoderType(int(bitstream[85:87].to01(), 2))
        assert bitstream[87:88].to01() == '0' # Reserved
        self.quantizer_fidelity_control_method = QuantizerFidelityControlMethod(int(bitstream[88:90].to01(), 2))
        assert bitstream[90:92].to01() == '00' # Reserved
        self.supplementary_information_table_count = int(bitstream[92:96].to01(), 2)

        bitstream = bitstream[96:]

        # Supplementary information tables
        if self.supplementary_information_table_count > 0:
            exit("Supplementary information tables not implemented")
        
            # Predictor metadata
        # Predictor primary structure
        assert bitstream[0:1].to01() == '0'
        self.sample_representative_flag = SampleRepresentativeFlag(int(bitstream[1:2].to01(), 2))
        self.prediction_bands_num = int(bitstream[2:6].to01(), 2)
        self.prediction_mode = PredictionMode(int(bitstream[6:7].to01(), 2))
        self.weight_exponent_offset_flag = WeightExponentOffsetFlag(int(bitstream[7:8].to01(), 2))
        self.local_sum_type = LocalSumType(int(bitstream[8:10].to01(), 2))
        self.register_size = int(bitstream[10:16].to01(), 2)
        self.weight_component_resolution = int(bitstream[16:20].to01(), 2)
        self.weight_update_change_interval = int(bitstream[20:24].to01(), 2)
        self.weight_update_initial_parameter = int(bitstream[24:28].to01(), 2)
        self.weight_update_final_parameter = int(bitstream[28:32].to01(), 2)
        self.weight_exponent_offset_table_flag = WeightExponentOffsetTableFlag(int(bitstream[32:33].to01(), 2))
        self.weight_init_method = WeightInitMethod(int(bitstream[33:34].to01(), 2))
        self.weight_init_table_flag = WeightInitTableFlag(int(bitstream[34:35].to01(), 2))
        self.weight_init_resolution = int(bitstream[35:40].to01(), 2)
        bitstream = bitstream[40:]

        # Weight tables subpart
        if self.weight_init_table_flag == WeightInitTableFlag.INCLUDED or \
            self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.INCLUDED:
            exit("Weight tables not implemented")

        # Predictor quantization structure
        if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.LOSSLESS:

            # Predictor quantization error limit update period structure
            if self.sample_encoding_order != SampleEncodingOrder.BSQ:
                assert bitstream[0:1].to01() == '0'
                self.periodic_error_updating_flag = PeriodicErrorUpdatingFlag(int(bitstream[1:2].to01(), 2))
                assert bitstream[2:4].to01() == '00'
                self.error_update_period_exponent = int(bitstream[4:8].to01(), 2)
                bitstream = bitstream[8:]
            else:
                self.periodic_error_updating_flag = PeriodicErrorUpdatingFlag.NOT_USED
                self.error_update_period_exponent = 0

            # Predictor quantization absolute error limit structure
            if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.RELATIVE_ONLY:
                assert bitstream[0:1].to01() == '0'
                self.absolute_error_limit_assignment_method = ErrorLimitAssignmentMethod(int(bitstream[1:2].to01(), 2))
                assert bitstream[2:4].to01() == '00'
                self.absolute_error_limit_bit_depth = int(bitstream[4:8].to01(), 2)
                self.absolute_error_limit_value = int(bitstream[8:8+self.absolute_error_limit_bit_depth].to01(), 2)
                bitstream = bitstream[8+self.absolute_error_limit_bit_depth:]

                if self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                    exit("Band-dependent absolute error limit not implemented")
            else:
                self.absolute_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_INDEPENDENT
                self.absolute_error_limit_bit_depth = 0
                self.absolute_error_limit_value = 0
            
            # Predictor quantization relative error limit structure
            if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
                assert bitstream[0:1].to01() == '0'
                self.relative_error_limit_assignment_method = ErrorLimitAssignmentMethod(int(bitstream[1:2].to01(), 2))
                assert bitstream[2:4].to01() == '00'
                self.relative_error_limit_bit_depth = int(bitstream[4:8].to01(), 2)
                self.relative_error_limit_value = int(bitstream[8:8+self.relative_error_limit_bit_depth].to01(), 2)
                bitstream = bitstream[8+self.relative_error_limit_bit_depth:]

                if self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                    exit("Band-dependent relative error limit not implemented")
            else:
                self.relative_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_INDEPENDENT
                self.relative_error_limit_bit_depth = 0
                self.relative_error_limit_value = 0
        else:
            self.periodic_error_updating_flag = PeriodicErrorUpdatingFlag.NOT_USED
            self.error_update_period_exponent = 0
            self.absolute_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_INDEPENDENT
            self.absolute_error_limit_bit_depth = 1
            self.absolute_error_limit_value = 0
            self.relative_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_INDEPENDENT
            self.relative_error_limit_bit_depth = 1
            self.relative_error_limit_value = 0

        # Predictor sample representative structure
        if self.sample_representative_flag == SampleRepresentativeFlag.INCLUDED:
            assert bitstream[0:5].to01() == '00000'
            self.sample_representative_resolution = int(bitstream[5:8].to01(), 2)
            assert bitstream[8:9].to01() == '0'
            self.band_varying_damping_flag = BandVaryingDampingFlag(int(bitstream[9:10].to01(), 2))
            self.damping_table_flag = DampingTableFlag(int(bitstream[10:11].to01(), 2))
            assert bitstream[11:12].to01() == '0'
            self.fixed_damping_value = int(bitstream[12:16].to01(), 2)
            assert bitstream[16:17].to01() == '0'
            self.band_varying_offset_flag = BandVaryingOffsetFlag(int(bitstream[17:18].to01(), 2))
            self.damping_offset_table_flag = OffsetTableFlag(int(bitstream[18:19].to01(), 2))
            assert bitstream[19:20].to01() == '0'
            self.fixed_offset_value = int(bitstream[20:24].to01(), 2)
            bitstream = bitstream[24:]

            # Damping and offset table sublocks
            if self.damping_table_flag == DampingTableFlag.INCLUDED or \
                self.damping_offset_table_flag == OffsetTableFlag.INCLUDED:
                exit("Damping tables not implemented")
        else:
            self.sample_representative_resolution = 0
            self.band_varying_damping_flag = BandVaryingDampingFlag.BAND_INDEPENDENT
            self.damping_table_flag = DampingTableFlag.NOT_INCLUDED
            self.fixed_damping_value = 0
            self.band_varying_offset_flag = BandVaryingOffsetFlag.BAND_INDEPENDENT
            self.damping_offset_table_flag = OffsetTableFlag.NOT_INCLUDED
            self.fixed_offset_value = 0
        
            # Entropy coder metadata
        # Sample-adaptive entropy coder
        if self.entropy_coder_type == EntropyCoderType.SAMPLE_ADAPTIVE:
            self.unary_length_limit = int(bitstream[0:5].to01(), 2)
            self.rescaling_counter_size = int(bitstream[5:8].to01(), 2)
            self.initial_count_exponent = int(bitstream[8:11].to01(), 2)
            self.accumulator_init_constant = int(bitstream[11:15].to01(), 2)
            self.accumulator_init_table_flag = AccumulatorInitTableFlag(int(bitstream[15:16].to01(), 2))
            bitstream = bitstream[16:]

            # Accumulator initialization table subblock
            if self.accumulator_init_table_flag == AccumulatorInitTableFlag.INCLUDED:
                exit("Accumulator initialization table not implemented")
        
        # Hybrid entropy coder
        elif self.entropy_coder_type == EntropyCoderType.HYBRID:
            self.unary_length_limit = int(bitstream[0:5].to01(), 2)
            self.rescaling_counter_size = int(bitstream[5:8].to01(), 2)
            self.initial_count_exponent = int(bitstream[8:11].to01(), 2)
            assert bitstream[11:16].to01() == '00000'
            bitstream = bitstream[16:]
        
        # Block-adaptive entropy coder
        elif self.entropy_coder_type == EntropyCoderType.BLOCK_ADAPTIVE:
            assert bitstream[0:1].to01() == '0'
            self.block_size = int(bitstream[1:3].to01(), 2)
            self.restricted_code_options_flag = RestrictedCodeOptionsFlag(int(bitstream[3:4].to01(), 2))
            self.reference_sample_interval = int(bitstream[4:16].to01(), 2)
            bitstream = bitstream[16:]

        assert len(bitstream) == 0
        self.__check_legal_config()

    
    def __check_legal_config(self):
        assert 0 <= self.user_defined_data and self.user_defined_data < 2**8
        assert 0 <= self.x_size and self.x_size < 2**16
        assert 0 <= self.y_size and self.y_size < 2**16
        assert 0 <= self.z_size and self.z_size < 2**16
        assert self.sample_type in SampleType
        assert self.large_d_flag in LargeDFlag
        assert 0 <= self.dynamic_range and self.dynamic_range < 16
        assert self.sample_encoding_order in SampleEncodingOrder
        assert (self.sub_frame_interleaving_depth == 0 and self.sample_encoding_order == SampleEncodingOrder.BSQ) or (0 <= self.sub_frame_interleaving_depth and self.sub_frame_interleaving_depth + 2**16 * (self.sub_frame_interleaving_depth == 0) <= self.z_size and self.sample_encoding_order == SampleEncodingOrder.BI)
        assert 0 <= self.output_word_size and self.output_word_size < 8
        assert self.entropy_coder_type in EntropyCoderType
        assert self.entropy_coder_type == EntropyCoderType.SAMPLE_ADAPTIVE # Other encoders not yet implemented
        assert self.quantizer_fidelity_control_method in QuantizerFidelityControlMethod
        assert 0 <= self.supplementary_information_table_count and self.supplementary_information_table_count <= 15
        assert self.supplementary_information_table_count == 0 # Not implemented

        assert self.sample_representative_flag in SampleRepresentativeFlag
        assert 0 <= self.prediction_bands_num and self.prediction_bands_num < 16
        assert self.prediction_mode in PredictionMode
        assert self.weight_exponent_offset_flag in WeightExponentOffsetFlag
        assert self.weight_exponent_offset_flag == WeightExponentOffsetFlag.ALL_ZERO # Table not implemented
        assert self.local_sum_type in LocalSumType
        assert max(32, self.get_dynamic_range_bits() + (self.weight_component_resolution + 4) + 2) <= self.register_size + 64 * int(self.register_size == 0) and self.register_size < 64
        assert 4 <= self.weight_component_resolution + 4 and self.weight_component_resolution + 4 <= 19
        assert 4 <= self.weight_update_change_interval + 4 and self.weight_update_change_interval + 4 <= 11
        assert -6 <= self.weight_update_initial_parameter - 6 and self.weight_update_initial_parameter - 6 <= 9
        assert -6 <= self.weight_update_final_parameter - 6 and self.weight_update_final_parameter - 6 <= 9
        assert self.weight_exponent_offset_table_flag in WeightExponentOffsetTableFlag
        assert self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.NOT_INCLUDED # Table not implemented
        assert self.weight_init_method in WeightInitMethod
        assert self.weight_init_table_flag in WeightInitTableFlag
        assert self.weight_init_table_flag == WeightInitTableFlag.NOT_INCLUDED # Table not implemented
        assert (self.weight_init_method == WeightInitMethod.CUSTOM and 3 <= self.weight_init_resolution and self.weight_init_resolution <= self.weight_component_resolution + 4 + 3) or (self.weight_init_method == WeightInitMethod.DEFAULT and self.weight_init_resolution == 0)

        assert self.periodic_error_updating_flag in PeriodicErrorUpdatingFlag
        assert self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED # Not implemented
        assert (self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED and 0 <= self.error_update_period_exponent and self.error_update_period_exponent <= 9) or (self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED and self.error_update_period_exponent == 0)
        assert self.error_update_period_exponent == 0 # Not implemented
        assert self.absolute_error_limit_assignment_method in ErrorLimitAssignmentMethod
        assert self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT # Not implemented
        assert 0 <= self.absolute_error_limit_bit_depth and self.absolute_error_limit_bit_depth + 16 * int(self.absolute_error_limit_bit_depth == 0) <= min(self.get_dynamic_range_bits() - 1, 16)
        assert 0 <= self.absolute_error_limit_value and self.absolute_error_limit_value <= 2**self.absolute_error_limit_bit_depth - 1
        assert self.relative_error_limit_assignment_method in ErrorLimitAssignmentMethod
        assert self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT # Not implemented
        assert 0 <= self.relative_error_limit_bit_depth and self.relative_error_limit_bit_depth + 16 * int(self.relative_error_limit_bit_depth == 0) <= min(self.get_dynamic_range_bits() - 1, 16)
        assert 0 <= self.relative_error_limit_value and self.relative_error_limit_value <= 2**self.relative_error_limit_bit_depth - 1

        assert 0 <= self.sample_representative_resolution and self.sample_representative_resolution <= 4
        assert self.band_varying_damping_flag in BandVaryingDampingFlag
        assert self.band_varying_damping_flag == BandVaryingDampingFlag.BAND_INDEPENDENT # Not implemented
        assert self.damping_table_flag in DampingTableFlag
        assert self.damping_table_flag == DampingTableFlag.NOT_INCLUDED # Table not implemented
        assert (self.damping_table_flag == DampingTableFlag.NOT_INCLUDED and 0 <= self.fixed_damping_value and self.fixed_damping_value <= 2**self.sample_representative_resolution - 1) or (self.damping_table_flag == DampingTableFlag.INCLUDED and self.fixed_damping_value == 0)
        assert self.band_varying_offset_flag in BandVaryingOffsetFlag
        assert self.band_varying_offset_flag == BandVaryingOffsetFlag.BAND_INDEPENDENT # Not implemented
        assert self.damping_offset_table_flag in OffsetTableFlag
        assert self.damping_offset_table_flag == OffsetTableFlag.NOT_INCLUDED # Table not implemented
        assert (self.damping_offset_table_flag == OffsetTableFlag.NOT_INCLUDED and 0 <= self.fixed_offset_value and self.fixed_offset_value <= 2**self.sample_representative_resolution - 1) or (self.damping_offset_table_flag == OffsetTableFlag.INCLUDED and self.fixed_offset_value == 0)

        assert (8 <= self.unary_length_limit and self.unary_length_limit < 32) or self.unary_length_limit == 0
        assert max(4, self.initial_count_exponent + 8 * int(self.initial_count_exponent == 0) - 1) <= self.rescaling_counter_size + 4 and self.rescaling_counter_size + 4 <= 11
        assert 0 <= self.initial_count_exponent and self.initial_count_exponent < 8
        assert (0 <= self.accumulator_init_constant and self.accumulator_init_constant <= min(self.get_dynamic_range_bits() - 2, 14)) or (self.accumulator_init_table_flag == AccumulatorInitTableFlag.INCLUDED and self.accumulator_init_constant == 15)
        assert self.accumulator_init_table_flag in AccumulatorInitTableFlag
        assert self.accumulator_init_table_flag == AccumulatorInitTableFlag.NOT_INCLUDED # Table not implemented
    
    def __encode_essential_subpart_structure(self):
        bitstream = bitarray()
        bitstream += bin(self.user_defined_data)[2:].zfill(8)
        bitstream += bin(self.x_size)[2:].zfill(16)
        bitstream += bin(self.y_size)[2:].zfill(16)
        bitstream += bin(self.z_size)[2:].zfill(16)
        bitstream += bin(self.sample_type.value)[2:].zfill(1)
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.large_d_flag.value)[2:].zfill(1)
        bitstream += bin(self.dynamic_range)[2:].zfill(4)
        bitstream += bin(self.sample_encoding_order.value)[2:].zfill(1)
        bitstream += bin(self.sub_frame_interleaving_depth)[2:].zfill(16)
        bitstream += 2 * '0' # Reserved
        bitstream += bin(self.output_word_size)[2:].zfill(3)
        bitstream += bin(self.entropy_coder_type.value)[2:].zfill(2)
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.quantizer_fidelity_control_method.value)[2:].zfill(2)
        bitstream += 2 * '0' # Reserved
        bitstream += bin(self.supplementary_information_table_count)[2:].zfill(4)
        assert len(bitstream) == 12 * 8
        return bitstream
    
    def __encode_predictor_primary_structure(self):
        bitstream = bitarray()
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.sample_representative_flag.value)[2:].zfill(1)
        bitstream += bin(self.prediction_bands_num)[2:].zfill(4)
        bitstream += bin(self.prediction_mode.value)[2:].zfill(1)
        bitstream += bin(self.weight_exponent_offset_flag.value)[2:].zfill(1)
        bitstream += bin(self.local_sum_type.value)[2:].zfill(2)
        bitstream += bin(self.register_size)[2:].zfill(6)
        bitstream += bin(self.weight_component_resolution)[2:].zfill(4)
        bitstream += bin(self.weight_update_change_interval)[2:].zfill(4)
        bitstream += bin(self.weight_update_initial_parameter)[2:].zfill(4)
        bitstream += bin(self.weight_update_final_parameter)[2:].zfill(4)
        bitstream += bin(self.weight_exponent_offset_table_flag.value)[2:].zfill(1)
        bitstream += bin(self.weight_init_method.value)[2:].zfill(1)
        bitstream += bin(self.weight_init_table_flag.value)[2:].zfill(1)
        bitstream += bin(self.weight_init_resolution)[2:].zfill(5)
        assert len(bitstream) == 8 * 5
        if self.weight_init_table_flag == WeightInitTableFlag.INCLUDED:
            exit("Weight initialization table not implemented")
        if self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.INCLUDED:
            exit("Weight exponent offset table not implemented")
        return bitstream
    
    def __encode_predictor_quantization_error_limit_update_period_structure(self):
        bitstream = bitarray()
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.periodic_error_updating_flag.value)[2:].zfill(1)
        bitstream += 2 * '0' # Reserved
        bitstream += bin(self.error_update_period_exponent)[2:].zfill(4)
        assert len(bitstream) == 8
        return bitstream
    
    def __encode_predictor_quantization_absolute_error_limit_structure(self):
        bitstream = bitarray()
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.absolute_error_limit_assignment_method.value)[2:].zfill(1)
        bitstream += 2 * '0' # Reserved
        bitstream += bin(self.absolute_error_limit_bit_depth)[2:].zfill(4)
        bitstream += bin(self.absolute_error_limit_value)[2:].zfill(self.absolute_error_limit_bit_depth + 16 * int(self.absolute_error_limit_bit_depth == 0))
        assert len(bitstream) == (self.absolute_error_limit_bit_depth + 16 * int(self.absolute_error_limit_bit_depth == 0)) + 8
        bitstream += (8 - (len(bitstream) % 8)) * '0'
        assert len(bitstream) % 8 == 0
        return bitstream
    
    def __encode_predictor_quantization_relative_error_limit_structure(self):
        bitstream = bitarray()
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.relative_error_limit_assignment_method.value)[2:].zfill(1)
        bitstream += 2 * '0' # Reserved
        bitstream += bin(self.relative_error_limit_bit_depth)[2:].zfill(4)
        bitstream += bin(self.relative_error_limit_value)[2:].zfill(self.relative_error_limit_bit_depth + 16 * int(self.relative_error_limit_bit_depth == 0))
        assert len(bitstream) == (self.relative_error_limit_bit_depth + 16 * int(self.relative_error_limit_bit_depth == 0)) + 8
        bitstream += (8 - (len(bitstream) % 8)) * '0'
        assert len(bitstream) % 8 == 0
        return bitstream
    
    def __encode_predictor_quantization_structure(self):
        bitstream = bitarray()
        if self.sample_encoding_order != SampleEncodingOrder.BSQ:
            bitstream += self.__encode_predictor_quantization_error_limit_update_period_structure()
        if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.RELATIVE_ONLY:
            bitstream += self.__encode_predictor_quantization_absolute_error_limit_structure()
        if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
            bitstream += self.__encode_predictor_quantization_relative_error_limit_structure()
        return bitstream
    
    def __encode_predictor_sample_representative_structure(self):
        bitstream = bitarray()
        bitstream += 5 * '0' # Reserved
        bitstream += bin(self.sample_representative_resolution)[2:].zfill(3)
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.band_varying_damping_flag.value)[2:].zfill(1)
        bitstream += bin(self.damping_table_flag.value)[2:].zfill(1)
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.fixed_damping_value)[2:].zfill(4)
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.band_varying_offset_flag.value)[2:].zfill(1)
        bitstream += bin(self.damping_offset_table_flag.value)[2:].zfill(1)
        bitstream += 1 * '0'
        bitstream += bin(self.fixed_offset_value)[2:].zfill(4)
        assert len(bitstream) == 8 * 3
        if self.damping_table_flag == DampingTableFlag.INCLUDED:
            exit("Damping table not implemented")
        if self.damping_offset_table_flag == OffsetTableFlag.INCLUDED:
            exit("Damping offset table not implemented")
        return bitstream

    def __encode_entropy_coder_sample_adaptive_structure(self):
        bitstream = bitarray()
        bitstream += bin(self.unary_length_limit)[2:].zfill(5)
        bitstream += bin(self.rescaling_counter_size)[2:].zfill(3)
        bitstream += bin(self.initial_count_exponent)[2:].zfill(3)
        bitstream += bin(self.accumulator_init_constant)[2:].zfill(4)
        bitstream += bin(self.accumulator_init_table_flag.value)[2:].zfill(1)
        assert len(bitstream) == 8 * 2
        if self.accumulator_init_table_flag == AccumulatorInitTableFlag.INCLUDED:
            exit("Accumulator initialization table not implemented")   
        return bitstream
    
    def __create_header_bitstream(self):
        bitstream = bitarray()

        bitstream += self.__encode_essential_subpart_structure()
        bitstream += self.__encode_predictor_primary_structure()

        if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.LOSSLESS:
            bitstream += self.__encode_predictor_quantization_structure()
        if self.sample_representative_flag == SampleRepresentativeFlag.INCLUDED:
            bitstream += self.__encode_predictor_sample_representative_structure()        
        if self.entropy_coder_type == EntropyCoderType.SAMPLE_ADAPTIVE:
            bitstream += self.__encode_entropy_coder_sample_adaptive_structure()
        
        self.header_bitstream = bitstream

    def set_encoding_order_bip(self):
        self.sample_encoding_order = SampleEncodingOrder.BI
        self.sub_frame_interleaving_depth = self.z_size

    def set_encoding_order_bil(self):
        self.sample_encoding_order = SampleEncodingOrder.BI
        self.sub_frame_interleaving_depth = 1
    
    def get_dynamic_range_bits(self):
        dynamic_range_bits = self.dynamic_range
        if dynamic_range_bits == 0:
            dynamic_range_bits = 16
        if self.large_d_flag == LargeDFlag.LARGE_D:
            dynamic_range_bits += 16
        return dynamic_range_bits
    
    def get_header_bitstream(self):
        self.__create_header_bitstream()
        return self.header_bitstream

    def save_header(self, output_folder):
        with open(output_folder + "/header.bin", "wb") as file:
            self.get_header_bitstream().tofile(file)
        