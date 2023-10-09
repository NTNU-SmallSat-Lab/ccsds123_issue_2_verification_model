from enum import Enum
import re

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


class Header:
    """
    Header class for storing image metadata and configuration options.
    """

    ################
    # Image metadata
    ################
    user_defined_data = 0
    x_size = 0 # N_x. 1<=N_x<=2^16-1
    y_size = 0 # N_y. 1<=N_y<=2^16-1
    z_size = 0 # N_z. 1<=N_z<=2^16-1
    sample_type = SampleType.UNSIGNED_INTEGER
    large_d_flag = LargeDFlag.SMALL_D
    dynamic_range = 0 # D. Encode as D%16. 1<=D<=32
    sample_encoding_order = SampleEncodingOrder.BI
    sub_frame_interleaving_depth = z_size # M. Encode as M%16. M=1 for BIL, M=z_size for BIP. 1<=M<=z_size
    output_word_size = 1 # B. Encode as B%8. 1<=B<=16
    entropy_coder_type = EntropyCoderType.HYBRID
    quantizer_fidelity_control_method = QuantizerFidelityControlMethod.ABSOLUTE_AND_RELATIVE
    supplementary_information_table_count = 0 # tau. 1<=tau<=15. Supplementary information tables are not implemented

    #####################
    # Predicator metadata
    #####################
    sample_representative_flag = SampleRepresentativeFlag.INCLUDED
    prediction_bands_num = 4 # P. 0<=P<=15
    prediction_mode = PredictionMode.FULL
    weight_exponent_offset_flag = WeightExponentOffsetFlag.NOT_ALL_ZERO
    local_sum_type = LocalSumType.WIDE_NEIGHBOR_ORIENTED
    register_size = 32 # R. Encode as R%64. max{32,D+Omega+2}<=R<=64
    weight_component_resolution = 13 # Omega. Encode as Omega-4. 4<=Omega<=19
    weight_update_change_interval = 6 # t_inc. Encode as log2(t_inc)-4. 2^4<=t_inc<=2^11
    weight_update_initial_parameter = -1 # nu_min. Encode as nu_min+6. -6<=nu_min<=nu_max<=9
    weight_update_final_parameter    = 3 # nu_max. Encode as nu_max+6. -6<=nu_min<=nu_max<=9
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
    absolute_error_limit_bit_depth = 15 # D_A. Encode as D_A%16. 1<=D_A<=min{D−1,16}
    absolute_error_limit_value = 2 # A*. 0<=A*<=2^D_A-1. TODO: Support BAND_DEPENDENT values
    # Relative error limit
    relative_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_INDEPENDENT
    relative_error_limit_bit_depth = 15 # D_R. Encode as D_R%16. 1<=D_R<=min{D−1,16}
    relative_error_limit_value = 2 # R*. 0<=R*<=2^D_R-1. TODO: Support BAND_DEPENDENT values

    # Sample Representative
    sample_representative_resolution = 4 # Theta. 0<=Theta<=4
    band_varying_damping_flag = BandVaryingDampingFlag.BAND_INDEPENDENT
    damping_table_flag = DampingTableFlag.NOT_INCLUDED
    fixed_damping_value = 8 # phi. Encode as 0 if damping_table_flag=INCLUDED, otherwise as phi. 0<=phi<=2^Theta-1
    band_varying_offset_flag = BandVaryingOffsetFlag.BAND_INDEPENDENT
    damping_offset_table_flag = OffsetTableFlag.NOT_INCLUDED
    fixed_offset_value = 8 # psi. Encode as 0 if damping_offset_table_flag=INCLUDED, otherwise as psi. 0<=psi<=2^Theta-1. psi=0 if lossless
    # TODO: Support damping table subblock
    # TODO: Support offset table subblock

    ########################
    # Entropy coder metadata
    ########################
    # Sample-adaptive entropy coder and Hybrid entropy coder
    unary_length_limit = 16 # U_max. Encode as U_max%32. 8<=L<=32
    rescaling_counter_size = 11 # gamma*. Encode as gamma*-4. Max{4,gamma_0+1}<=γ<=11
    initial_count_exponent = 5 # gamma_0. 1<=gamma_0<=8
    # Remaining sample-adaptive entropy coder
    accumulator_init_table_flag = AccumulatorInitTableFlag.NOT_INCLUDED
    # TODO: Support accumulator initialization table

    # Block-adaptive entropy coder
    # TODO: Support block-adaptive entropy coder


    def __init__(self, image_name):
        self.set_config_according_to_image_name(image_name)
        
    def set_config_according_to_image_name(self, image_name):
        # TODO: Actually learn regex and do this properly
        self.x_size = int(re.findall('x(.*).raw', image_name)[0].split("x")[-1]) 
        self.y_size = int(re.findall('x(.+)x', image_name)[0])
        self.z_size = int(re.findall('-(.+)x', re.findall('-(.+)x', image_name)[0])[0])
        format = re.findall('-(.*)-', image_name)[0]
        self.sample_type = SampleType.UNSIGNED_INTEGER if format[0] == 'u' else SampleType.SIGNED_INTEGER
        self.large_d_flag = LargeDFlag.SMALL_D if int(format[1:3]) <= 16 else LargeDFlag.LARGE_D
        self.dynamic_range = int(format[1:3]) % 16
        if format[3:5] != 'be':
            exit("Only big endian is supported")
    