from enum import Enum
import re
import numpy as np
from bitarray import bitarray
from math import ceil

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

class TableType(Enum):
    UNSIGNED_INTEGER = 0
    SIGNED_INTEGER = 1
    FLOAT = 2

class TableStructure(Enum):
    ZERO_DIMENSIONAL = 0
    ONE_DIMENSIONAL = 1
    TWO_DIMENSIONAL_ZX = 2
    TWO_DIMENSIONAL_YX = 3

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

class SupplementaryInformationTable:
    table_type = TableType.UNSIGNED_INTEGER
    table_purpose = 0
    table_structure = TableStructure.ZERO_DIMENSIONAL
    user_defined_data = 0
    
    table_data_subblock = bitarray()

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
    dynamic_range = 10 # D. Encode as D%16. 1<=D<=32
    sample_encoding_order = SampleEncodingOrder.BI
    sub_frame_interleaving_depth = 1 # M. Encode as M%2^16. M=1 for BIL, M=z_size for BIP. 1<=M<=z_size
    output_word_size = 1 # B. Encode as B%8. 1<=B<=8
    entropy_coder_type = EntropyCoderType.SAMPLE_ADAPTIVE
    quantizer_fidelity_control_method = QuantizerFidelityControlMethod.ABSOLUTE_AND_RELATIVE
    supplementary_information_table_count = 0 # tau. 0<=tau<=15
    supplementary_information_tables = []

    #####################
    # Predicator metadata
    #####################
    sample_representative_flag = SampleRepresentativeFlag.INCLUDED
    prediction_bands_num = 1 # P. 0<=P<=15
    prediction_mode = PredictionMode.REDUCED
    weight_exponent_offset_flag = WeightExponentOffsetFlag.ALL_ZERO
    local_sum_type = LocalSumType.WIDE_NEIGHBOR_ORIENTED
    register_size = 0 # R. Encode as R%64. max{32,D+Omega+2}<=R<=64
    weight_component_resolution = 15 # Omega. Encode as Omega-4. 4<=Omega<=19
    weight_update_change_interval = 6 # t_inc. Encode as log2(t_inc)-4. 2^4<=t_inc<=2^11
    weight_update_initial_parameter = 6 # nu_min. Encode as nu_min+6. -6<=nu_min<=nu_max<=9
    weight_update_final_parameter = 10 # nu_max. Encode as nu_max+6. -6<=nu_min<=nu_max<=9
    weight_exponent_offset_table_flag = WeightExponentOffsetTableFlag.NOT_INCLUDED
    weight_init_method = WeightInitMethod.DEFAULT
    weight_init_table_flag = WeightInitTableFlag.NOT_INCLUDED
    weight_init_resolution = 0 # Q. Encode as 0 if weight_init_method=DEFAULT, otherwise as Q. 3<=Q<=Omega+3
    
    # Weight initialization table
    weight_init_table_value = 0 # The default value the weight initialization table cells are initialized to. Not part of standard
    weight_init_table = None # Lambda. Array of size N_z * C_z

    # Weight exponent offset table
    weight_exponent_offset_value = 0 # The default value the weight exponent offset table cells are initialized to. Not part of standard
    weight_exponent_offset_table = None # sigma (in word-final position). Array of size N_z * C_z

    # Quantization
    # Error limit update
    periodic_error_updating_flag = PeriodicErrorUpdatingFlag.USED
    error_update_period_exponent = 0 # u. Encode as 0 if periodic_error_updating_flag=NOT_USED, otherwise as u. 0<=u<=9
    periodic_absolute_error_limit_table = None
    periodic_relative_error_limit_table = None
    # Absolute error limit
    absolute_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_DEPENDENT
    absolute_error_limit_bit_depth = 5 # D_A. Encode as D_A%16. 1<=D_A<=min{D − 1,16}
    absolute_error_limit_value = 2 # A*. 0<=A*<=2^D_A-1.
    absolute_error_limit_table = None # a_z. Array of size N_z
    # Relative error limit
    relative_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_DEPENDENT
    relative_error_limit_bit_depth = 9 # D_R. Encode as D_R%16. 1<=D_R<=min{D − 1,16}
    relative_error_limit_value = 20 # R*. 0<=R*<=2^D_R-1.
    relative_error_limit_table = None # r_z. Array of size N_z

    # Sample Representative
    sample_representative_resolution = 4 # Theta. 0<=Theta<=4
    band_varying_damping_flag = BandVaryingDampingFlag.BAND_DEPENDENT
    damping_table_flag = DampingTableFlag.INCLUDED
    fixed_damping_value = 0 # phi. Encode as 0 if damping_table_flag=INCLUDED, otherwise as phi. 0<=phi<=2^Theta-1
    band_varying_offset_flag = BandVaryingOffsetFlag.BAND_DEPENDENT
    damping_offset_table_flag = OffsetTableFlag.INCLUDED
    fixed_offset_value = 0 # psi. Encode as 0 if damping_offset_table_flag=INCLUDED, otherwise as psi. 0<=psi<=2^Theta-1. psi=0 if lossless

    damping_table_array = None # phi_z. Array of size N_z
    damping_offset_table_array = None # psi_z. Array of size N_z

    ########################
    # Entropy coder metadata
    ########################
    # Sample-adaptive entropy coder and Hybrid entropy coder
    unary_length_limit = 16 # U_max. Encode as U_max%32. 8<=U_max<=32
    rescaling_counter_size = 5 # gamma*. Encode as gamma*-4. Max{4,gamma_0+1}<=γ<=11
    initial_count_exponent = 5 # gamma_0. Encode as gamma_0%8. 1<=gamma_0<=8
    # Remaining sample-adaptive entropy coder
    accumulator_init_constant = 15 # K. Encode as 15 if K is not used. 0<=K<=min(D-2,14)
    accumulator_init_table_flag = AccumulatorInitTableFlag.NOT_INCLUDED
    
    accumulator_init_table = None # k''_z. Array of size N_z
    

    # Block-adaptive entropy coder
    block_size = 2 # B. 0: J=8, 1: J=16, 2: J=32, 3: J=64
    restricted_code_options_flag = RestrictedCodeOptionsFlag.UNRESTRICTED
    reference_sample_interval = 0 # r. Encode as r%2**12.

    header_bitstream = None
    optional_tables_bitstream = None

    def __init__(self, image_name):
        self.__set_config_according_to_image_name(image_name)

        if self.weight_init_method == WeightInitMethod.CUSTOM:
            self.set_weight_init_table_array_to_default()
        if self.weight_exponent_offset_flag == WeightExponentOffsetFlag.NOT_ALL_ZERO:
            self.set_weight_exponent_offset_table_array_to_default()
        if self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.ABSOLUTE_ONLY or \
            self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.ABSOLUTE_AND_RELATIVE:
            if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED:
                self.set_absolute_error_limit_table_array_to_default()
            elif self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED:
                self.set_periodic_absolute_error_limit_table_array_to_default()
        if self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.RELATIVE_ONLY or \
            self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.ABSOLUTE_AND_RELATIVE:
            if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED:
                self.set_relative_error_limit_table_array_to_default()
            elif self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED:
                self.set_periodic_relative_error_limit_table_array_to_default()
        self.set_damping_table_array_to_default()
        self.set_damping_offset_table_array_to_default()
        self.set_accumulator_init_table_to_default()
        
        self.__check_legal_config()
        
    def __set_config_according_to_image_name(self, image_name):
        self.x_size = int(re.findall('x(.*).raw', image_name)[0].split("x")[-1]) 
        self.y_size = int(re.findall('x(.+)x', image_name)[0])
        self.z_size = int(image_name.split('x')[0].split('-')[-1])
        format = re.findall('-(.*)-', image_name)[0]
        format = image_name.split('-')[-2].split('-')[-1]
        self.sample_type = SampleType.UNSIGNED_INTEGER if format[0] == 'u' else SampleType.SIGNED_INTEGER

    def __init_weight_init_table_array(self):
        weight_init_table_shape = (self.z_size + 2**16 * int(self.z_size == 0), self.prediction_bands_num + 3 * int(self.prediction_mode == PredictionMode.FULL))
        self.weight_init_table = np.zeros(weight_init_table_shape, dtype=np.int64)
    
    def __init_weight_exponent_offset_table_array(self):
        weight_exponent_offset_table_shape = (self.z_size + 2**16 * int(self.z_size == 0), self.prediction_bands_num + int(self.prediction_mode == PredictionMode.FULL))
        self.weight_exponent_offset_table = np.zeros(weight_exponent_offset_table_shape, dtype=np.int64)

    def __init_periodic_absolute_error_limit_table_array(self):
        periodic_absolute_error_limit_table_shape = (ceil((self.y_size + 2**16 * int(self.y_size == 0)) / 2**self.error_update_period_exponent), self.z_size + 2**16 * int(self.z_size == 0))
        self.periodic_absolute_error_limit_table = np.zeros(periodic_absolute_error_limit_table_shape, dtype=np.int64)

    def __init_periodic_relative_error_limit_table_array(self):
        periodic_relative_error_limit_table_shape = (ceil((self.y_size + 2**16 * int(self.y_size == 0)) / 2**self.error_update_period_exponent), self.z_size + 2**16 * int(self.z_size == 0))
        self.periodic_relative_error_limit_table = np.zeros(periodic_relative_error_limit_table_shape, dtype=np.int64)

    def __init_absolute_error_limit_table_array(self):
        self.absolute_error_limit_table = np.zeros(self.z_size + 2**16 * int(self.z_size == 0), dtype=np.int64)
    
    def __init_relative_error_limit_table_array(self):
        self.relative_error_limit_table = np.zeros(self.z_size + 2**16 * int(self.z_size == 0), dtype=np.int64)

    def __init_damping_table_array(self):
        self.damping_table_array = np.zeros(self.z_size + 2**16 * int(self.z_size == 0), dtype=np.int64)
    
    def __init_damping_offset_table_array(self):
        self.damping_offset_table_array = np.zeros(self.z_size + 2**16 * int(self.z_size == 0), dtype=np.int64)
    
    def __init_accumulator_init_table(self):
        self.accumulator_init_table = np.zeros(self.z_size + 2**16 * int(self.z_size == 0), dtype=np.int64)
    
    def set_config_from_file(self, header_file_location, optional_tables_file_location=None, error_limits_file_location=None):
        header_file = bitarray()
        optional_tables_file = bitarray()
        
        with open(header_file_location, "rb") as file:
            header_file.fromfile(file)

        if optional_tables_file_location is not None:
            with open(optional_tables_file_location, "rb") as file:
                optional_tables_file.fromfile(file)
        else:
            optional_tables_file = bitarray() # Empty bitarray
        
        assert len(header_file) % 8 == 0
        assert len(optional_tables_file) % 8 == 0
        
            # Image metadata 
        # Essential subpart
        self.user_defined_data = int(header_file[0:8].to01(), 2)
        self.x_size = int(header_file[8:24].to01(), 2)
        self.y_size = int(header_file[24:40].to01(), 2)
        self.z_size = int(header_file[40:56].to01(), 2)
        self.sample_type = SampleType(int(header_file[56:57].to01(), 2))
        assert header_file[57:58].to01() == '0' # Reserved
        self.large_d_flag = LargeDFlag(int(header_file[58:59].to01(), 2))
        self.dynamic_range = int(header_file[59:63].to01(), 2)
        self.sample_encoding_order = SampleEncodingOrder(int(header_file[63:64].to01(), 2))
        self.sub_frame_interleaving_depth = int(header_file[64:80].to01(), 2)
        assert header_file[80:82].to01() == '00' # Reserved
        self.output_word_size = int(header_file[82:85].to01(), 2)
        self.entropy_coder_type = EntropyCoderType(int(header_file[85:87].to01(), 2))
        assert header_file[87:88].to01() == '0' # Reserved
        self.quantizer_fidelity_control_method = QuantizerFidelityControlMethod(int(header_file[88:90].to01(), 2))
        assert header_file[90:92].to01() == '00' # Reserved
        self.supplementary_information_table_count = int(header_file[92:96].to01(), 2)

        header_file = header_file[96:]
        assert len(header_file) % 8 == 0
        assert len(optional_tables_file) % 8 == 0

        # Supplementary information tables
        self.supplementary_information_tables = [SupplementaryInformationTable() for i in range(self.supplementary_information_table_count)]
        for i in range(self.supplementary_information_table_count):
            self.supplementary_information_tables[i].table_type = TableType(int(header_file[0:2].to01(), 2))
            assert header_file[2:4].to01() == '00' # Reserved
            self.supplementary_information_tables[i].table_purpose = int(header_file[4:8].to01(), 2)
            assert header_file[8:9].to01() == '0' # Reserved
            self.supplementary_information_tables[i].table_structure = TableStructure(int(header_file[9:11].to01(), 2))
            assert header_file[11:12].to01() == '0' # Reserved
            self.supplementary_information_tables[i].user_defined_data = int(header_file[12:16].to01(), 2)
            header_file = header_file[16:]

            table_size = 0
            if self.supplementary_information_tables[i].table_structure == TableStructure.ZERO_DIMENSIONAL:
                table_size = 1
            elif self.supplementary_information_tables[i].table_structure == TableStructure.ONE_DIMENSIONAL:
                table_size = self.z_size + 2**16 * int(self.z_size == 0)
            elif self.supplementary_information_tables[i].table_structure == TableStructure.TWO_DIMENSIONAL_ZX:
                table_size = (self.z_size + 2**16 * int(self.z_size == 0)) * (self.x_size + 2**16 * int(self.x_size == 0))
            elif self.supplementary_information_tables[i].table_structure == TableStructure.TWO_DIMENSIONAL_YX:
                table_size = (self.y_size + 2**16 * int(self.y_size == 0)) * (self.x_size + 2**16 * int(self.x_size == 0))

            data_subblock_bits = 0
            if self.supplementary_information_tables[i].table_type != TableType.FLOAT:
                bit_depth = int(header_file[0:5].to01(), 2)
                bit_depth += 2**5 * int(bit_depth == 0)
                data_subblock_bits = bit_depth * table_size + 5
                
                
            elif self.supplementary_information_tables[i].table_type == TableType.FLOAT:
                significand_bit_depth = int(header_file[0:5].to01(), 2)
                exponent_bit_depth = int(header_file[5:8].to01(), 2)
                bit_depth = significand_bit_depth + exponent_bit_depth + 8 * int(exponent_bit_depth == 0) + 1
                data_subblock_bits = bit_depth * table_size + 8 + exponent_bit_depth + 8 * int(exponent_bit_depth == 0) # Exponent bias is included
            
            data_subblock_bits += (8 - data_subblock_bits % 8) % 8 # Add fill bits
            self.supplementary_information_tables[i].table_data_subblock = header_file[0:data_subblock_bits]
            header_file = header_file[data_subblock_bits:]
        
        assert len(header_file) % 8 == 0
        assert len(optional_tables_file) % 8 == 0

            # Predictor metadata
        # Predictor primary structure
        assert header_file[0:1].to01() == '0'
        self.sample_representative_flag = SampleRepresentativeFlag(int(header_file[1:2].to01(), 2))
        self.prediction_bands_num = int(header_file[2:6].to01(), 2)
        self.prediction_mode = PredictionMode(int(header_file[6:7].to01(), 2))
        self.weight_exponent_offset_flag = WeightExponentOffsetFlag(int(header_file[7:8].to01(), 2))
        self.local_sum_type = LocalSumType(int(header_file[8:10].to01(), 2))
        self.register_size = int(header_file[10:16].to01(), 2)
        self.weight_component_resolution = int(header_file[16:20].to01(), 2)
        self.weight_update_change_interval = int(header_file[20:24].to01(), 2)
        self.weight_update_initial_parameter = int(header_file[24:28].to01(), 2)
        self.weight_update_final_parameter = int(header_file[28:32].to01(), 2)
        self.weight_exponent_offset_table_flag = WeightExponentOffsetTableFlag(int(header_file[32:33].to01(), 2))
        self.weight_init_method = WeightInitMethod(int(header_file[33:34].to01(), 2))
        self.weight_init_table_flag = WeightInitTableFlag(int(header_file[34:35].to01(), 2))
        self.weight_init_resolution = int(header_file[35:40].to01(), 2)
        header_file = header_file[40:]

        assert len(header_file) % 8 == 0
        assert len(optional_tables_file) % 8 == 0
        
        # Weight tables subpart
        if self.weight_init_method == WeightInitMethod.CUSTOM:
            self.__init_weight_init_table_array()
            for z in range(self.weight_init_table.shape[0]):
                for j in range(min(z, self.prediction_bands_num) + 3 * int(self.prediction_mode == PredictionMode.FULL)):
                    if self.weight_init_table_flag == WeightInitTableFlag.INCLUDED:
                        number = header_file[0:self.weight_init_resolution].to01() # extract 4 bit two's complement number
                        self.weight_init_table[z, j] = int(number[0] == '1') * -2**(self.weight_init_resolution - 1) + int(number[1:], 2)
                        header_file = header_file[self.weight_init_resolution:]
                    elif self.weight_init_table_flag == WeightInitTableFlag.NOT_INCLUDED:
                        number = optional_tables_file[0:self.weight_init_resolution].to01() # extract 4 bit two's complement number
                        self.weight_init_table[z, j] = int(number[0] == '1') * -2**(self.weight_init_resolution - 1) + int(number[1:], 2)
                        optional_tables_file = optional_tables_file[self.weight_init_resolution:]
            # Skip fill bits
            if self.weight_init_table_flag == WeightInitTableFlag.INCLUDED:
                header_file = header_file[len(header_file) % 8:] 
            elif self.weight_init_table_flag == WeightInitTableFlag.NOT_INCLUDED:
                optional_tables_file = optional_tables_file[len(optional_tables_file) % 8:] 
        
        assert len(header_file) % 8 == 0
        assert len(optional_tables_file) % 8 == 0

        if self.weight_exponent_offset_flag == WeightExponentOffsetFlag.NOT_ALL_ZERO:
            self.__init_weight_exponent_offset_table_array()
            for z in range(self.weight_exponent_offset_table.shape[0]):
                for j in range(min(z, self.prediction_bands_num) + int(self.prediction_mode == PredictionMode.FULL)):
                    if self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.INCLUDED:
                        number = header_file[0:4].to01() # extract 4 bit two's complement number
                        self.weight_exponent_offset_table[z, j] = int(number[0] == '1') * -2**3 + int(number[1:4], 2) 
                        header_file = header_file[4:]
                    elif self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.NOT_INCLUDED:
                        number = optional_tables_file[0:4].to01() # extract 4 bit two's complement number
                        self.weight_exponent_offset_table[z, j] = int(number[0] == '1') * -2**3 + int(number[1:4], 2) 
                        optional_tables_file = optional_tables_file[4:]
            # Skip fill bits
            if self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.INCLUDED:
                header_file = header_file[len(header_file) % 8:] 
            elif self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.NOT_INCLUDED:
                optional_tables_file = optional_tables_file[len(optional_tables_file) % 8:]
        
        assert len(header_file) % 8 == 0
        assert len(optional_tables_file) % 8 == 0

        # Predictor quantization structure
        if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.LOSSLESS:

            # Predictor quantization error limit update period structure
            if self.sample_encoding_order != SampleEncodingOrder.BSQ:
                assert header_file[0:1].to01() == '0'
                self.periodic_error_updating_flag = PeriodicErrorUpdatingFlag(int(header_file[1:2].to01(), 2))
                assert header_file[2:4].to01() == '00'
                self.error_update_period_exponent = int(header_file[4:8].to01(), 2)
                header_file = header_file[8:]
            else:
                self.periodic_error_updating_flag = PeriodicErrorUpdatingFlag.NOT_USED
                self.error_update_period_exponent = 0
            
            assert len(header_file) % 8 == 0
            assert len(optional_tables_file) % 8 == 0

            # Predictor quantization absolute error limit structure
            if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.RELATIVE_ONLY:
                assert header_file[0:1].to01() == '0'
                self.absolute_error_limit_assignment_method = ErrorLimitAssignmentMethod(int(header_file[1:2].to01(), 2))
                assert header_file[2:4].to01() == '00'
                self.absolute_error_limit_bit_depth = int(header_file[4:8].to01(), 2)
                header_file = header_file[8:]

                if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED:
                    if self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                        self.absolute_error_limit_value = int(header_file[:self.get_absolute_error_limit_bit_depth_value()].to01(), 2)
                        self.set_absolute_error_limit_table_array_to_default()
                        header_file = header_file[self.get_absolute_error_limit_bit_depth_value():]
                    elif self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                        self.__init_absolute_error_limit_table_array()
                        for z in range(self.absolute_error_limit_table.shape[0]):
                            self.absolute_error_limit_table[z] = int(header_file[:self.get_absolute_error_limit_bit_depth_value()].to01(), 2)
                            header_file = header_file[self.get_absolute_error_limit_bit_depth_value():]               
                    
                    header_file = header_file[len(header_file) % 8:] # Skip fill bits

            else:
                self.absolute_error_limit_assignment_method = ErrorLimitAssignmentMethod.BAND_INDEPENDENT
                self.absolute_error_limit_bit_depth = 0
                self.absolute_error_limit_value = 0
            
            assert len(header_file) % 8 == 0
            assert len(optional_tables_file) % 8 == 0
            
            # Predictor quantization relative error limit structure
            if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
                assert header_file[0:1].to01() == '0'
                self.relative_error_limit_assignment_method = ErrorLimitAssignmentMethod(int(header_file[1:2].to01(), 2))
                assert header_file[2:4].to01() == '00'
                self.relative_error_limit_bit_depth = int(header_file[4:8].to01(), 2)
                header_file = header_file[8:]

                if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED:
                    if self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                        self.relative_error_limit_value = int(header_file[:self.get_relative_error_limit_bit_depth_value()].to01(), 2)
                        self.set_relative_error_limit_table_array_to_default()
                        header_file = header_file[self.get_relative_error_limit_bit_depth_value():]
                    elif self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                        self.__init_relative_error_limit_table_array()
                        for z in range(self.relative_error_limit_table.shape[0]):
                            self.relative_error_limit_table[z] = int(header_file[:self.get_relative_error_limit_bit_depth_value()].to01(), 2)
                            header_file = header_file[self.get_relative_error_limit_bit_depth_value():]

                    header_file = header_file[len(header_file) % 8:] # Skip fill bits

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

        assert len(header_file) % 8 == 0
        assert len(optional_tables_file) % 8 == 0

        # Predictor sample representative structure
        if self.sample_representative_flag == SampleRepresentativeFlag.INCLUDED:
            assert header_file[0:5].to01() == '00000'
            self.sample_representative_resolution = int(header_file[5:8].to01(), 2)
            assert header_file[8:9].to01() == '0'
            self.band_varying_damping_flag = BandVaryingDampingFlag(int(header_file[9:10].to01(), 2))
            self.damping_table_flag = DampingTableFlag(int(header_file[10:11].to01(), 2))
            assert header_file[11:12].to01() == '0'
            self.fixed_damping_value = int(header_file[12:16].to01(), 2)
            assert header_file[16:17].to01() == '0'
            self.band_varying_offset_flag = BandVaryingOffsetFlag(int(header_file[17:18].to01(), 2))
            self.damping_offset_table_flag = OffsetTableFlag(int(header_file[18:19].to01(), 2))
            assert header_file[19:20].to01() == '0'
            self.fixed_offset_value = int(header_file[20:24].to01(), 2)
            header_file = header_file[24:]

            assert len(header_file) % 8 == 0
            assert len(optional_tables_file) % 8 == 0

            # Damping and offset table sublocks
            if self.band_varying_damping_flag == BandVaryingDampingFlag.BAND_DEPENDENT:
                self.__init_damping_table_array()
                for i in range(self.damping_table_array.shape[0]):
                    if self.damping_table_flag == DampingTableFlag.INCLUDED:
                        self.damping_table_array[i] = int(header_file[:self.sample_representative_resolution].to01(), 2)
                        header_file = header_file[self.sample_representative_resolution:]
                    elif self.damping_table_flag == DampingTableFlag.NOT_INCLUDED:
                        self.damping_table_array[i] = int(optional_tables_file[:self.sample_representative_resolution].to01(), 2)
                        optional_tables_file = optional_tables_file[self.sample_representative_resolution:]
                if self.damping_table_flag == DampingTableFlag.INCLUDED:
                    header_file = header_file[len(header_file) % 8:]
                elif self.damping_table_flag == DampingTableFlag.NOT_INCLUDED:
                    optional_tables_file = optional_tables_file[len(optional_tables_file) % 8:]
            elif self.band_varying_damping_flag == BandVaryingDampingFlag.BAND_INDEPENDENT:
                self.set_damping_table_array_to_default()
            
            assert len(header_file) % 8 == 0
            assert len(optional_tables_file) % 8 == 0
            
            if self.band_varying_offset_flag == BandVaryingOffsetFlag.BAND_DEPENDENT:
                self.__init_damping_offset_table_array()
                for i in range(self.damping_offset_table_array.shape[0]):
                    if self.damping_offset_table_flag == OffsetTableFlag.INCLUDED:
                        self.damping_offset_table_array[i] = int(header_file[:self.sample_representative_resolution].to01(), 2)
                        header_file = header_file[self.sample_representative_resolution:]
                    elif self.damping_offset_table_flag == OffsetTableFlag.NOT_INCLUDED:
                        self.damping_offset_table_array[i] = int(optional_tables_file[:self.sample_representative_resolution].to01(), 2)
                        optional_tables_file = optional_tables_file[self.sample_representative_resolution:]
                if self.damping_offset_table_flag == OffsetTableFlag.INCLUDED:
                    header_file = header_file[len(header_file) % 8:]
                elif self.damping_offset_table_flag == OffsetTableFlag.NOT_INCLUDED:
                    optional_tables_file = optional_tables_file[len(optional_tables_file) % 8:]
            elif self.band_varying_offset_flag == BandVaryingOffsetFlag.BAND_INDEPENDENT:
                self.set_damping_offset_table_array_to_default()

        else:
            self.sample_representative_resolution = 0
            self.band_varying_damping_flag = BandVaryingDampingFlag.BAND_INDEPENDENT
            self.damping_table_flag = DampingTableFlag.NOT_INCLUDED
            self.fixed_damping_value = 0
            self.band_varying_offset_flag = BandVaryingOffsetFlag.BAND_INDEPENDENT
            self.damping_offset_table_flag = OffsetTableFlag.NOT_INCLUDED
            self.fixed_offset_value = 0
            self.set_damping_table_array_to_default()
            self.set_damping_offset_table_array_to_default()
        
        assert len(header_file) % 8 == 0
        assert len(optional_tables_file) % 8 == 0
        
            # Entropy coder metadata
        # Sample-adaptive entropy coder
        if self.entropy_coder_type == EntropyCoderType.SAMPLE_ADAPTIVE:
            self.unary_length_limit = int(header_file[0:5].to01(), 2)
            self.rescaling_counter_size = int(header_file[5:8].to01(), 2)
            self.initial_count_exponent = int(header_file[8:11].to01(), 2)
            self.accumulator_init_constant = int(header_file[11:15].to01(), 2)
            self.accumulator_init_table_flag = AccumulatorInitTableFlag(int(header_file[15:16].to01(), 2))
            header_file = header_file[16:]

            # Accumulator initialization table subblock
            if self.accumulator_init_constant == 15:
                self.__init_accumulator_init_table()
                for z in range(self.accumulator_init_table.shape[0]):
                    if self.accumulator_init_table_flag == AccumulatorInitTableFlag.INCLUDED:
                        self.accumulator_init_table[z] = int(header_file[:4].to01(), 2)
                        header_file = header_file[4:]
                    elif self.accumulator_init_table_flag == AccumulatorInitTableFlag.NOT_INCLUDED:
                        self.accumulator_init_table[z] = int(optional_tables_file[:4].to01(), 2)
                        optional_tables_file = optional_tables_file[4:]
                if self.accumulator_init_table_flag == AccumulatorInitTableFlag.INCLUDED:
                    header_file = header_file[len(header_file) % 8:]
                elif self.accumulator_init_table_flag == AccumulatorInitTableFlag.NOT_INCLUDED:
                    optional_tables_file = optional_tables_file[len(optional_tables_file) % 8:]
            else:
                self.set_accumulator_init_table_to_default()
        
        # Hybrid entropy coder
        elif self.entropy_coder_type == EntropyCoderType.HYBRID:
            self.unary_length_limit = int(header_file[0:5].to01(), 2)
            self.rescaling_counter_size = int(header_file[5:8].to01(), 2)
            self.initial_count_exponent = int(header_file[8:11].to01(), 2)
            assert header_file[11:16].to01() == '00000'
            header_file = header_file[16:]
        
        # Block-adaptive entropy coder
        elif self.entropy_coder_type == EntropyCoderType.BLOCK_ADAPTIVE:
            assert header_file[0:1].to01() == '0'
            self.block_size = int(header_file[1:3].to01(), 2)
            self.restricted_code_options_flag = RestrictedCodeOptionsFlag(int(header_file[3:4].to01(), 2))
            self.reference_sample_interval = int(header_file[4:16].to01(), 2)
            header_file = header_file[16:]
        
        assert len(header_file) == 0
        assert len(optional_tables_file) == 0

        # Read error limit file for periodic error limit updating
        if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED:
            error_limits_file = bitarray()
            if error_limits_file is not None:
                with open(error_limits_file_location, "rb") as file:
                    error_limits_file.fromfile(file)
            else:
                exit("Error limits file not provided")

            if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.RELATIVE_ONLY:
                self.__init_periodic_absolute_error_limit_table_array()
            if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
                self.__init_periodic_relative_error_limit_table_array()
            
            for i in range(ceil((self.y_size + 2**16 * int(self.y_size == 0)) / 2**self.error_update_period_exponent)):

                if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.RELATIVE_ONLY:
                    if self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                        self.periodic_absolute_error_limit_table[i,:] = int(error_limits_file[:16].to01(), 2)
                        error_limits_file = error_limits_file[16:]
                    elif self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                        for z in range(self.periodic_absolute_error_limit_table.shape[1]):
                            self.periodic_absolute_error_limit_table[i,z] = int(error_limits_file[:16].to01(), 2)
                            error_limits_file = error_limits_file[16:]

                if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
                    if self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                        self.periodic_relative_error_limit_table[i,:] = int(error_limits_file[:16].to01(), 2)
                        error_limits_file = error_limits_file[16:]
                    elif self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                        for z in range(self.periodic_relative_error_limit_table.shape[1]):
                            self.periodic_relative_error_limit_table[i,z] = int(error_limits_file[:16].to01(), 2)
                            error_limits_file = error_limits_file[16:]

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
        assert self.quantizer_fidelity_control_method in QuantizerFidelityControlMethod
        assert 0 <= self.supplementary_information_table_count and self.supplementary_information_table_count <= 15
        assert self.supplementary_information_table_count == len(self.supplementary_information_tables)

        assert self.sample_representative_flag in SampleRepresentativeFlag
        assert 0 <= self.prediction_bands_num and self.prediction_bands_num < 16
        assert self.prediction_mode in PredictionMode
        assert self.x_size != 1 or self.prediction_mode == PredictionMode.REDUCED # Can't use full prediction mode if x_size = 1. See standard section 4.3.1
        assert self.weight_exponent_offset_flag in WeightExponentOffsetFlag
        assert self.local_sum_type in LocalSumType
        assert max(32, self.get_dynamic_range_bits() + (self.weight_component_resolution + 4) + 2) <= self.register_size + 64 * int(self.register_size == 0) and self.register_size < 64
        assert 4 <= self.weight_component_resolution + 4 and self.weight_component_resolution + 4 <= 19
        assert 4 <= self.weight_update_change_interval + 4 and self.weight_update_change_interval + 4 <= 11
        assert -6 <= self.weight_update_initial_parameter - 6 and self.weight_update_initial_parameter - 6 <= 9
        assert -6 <= self.weight_update_final_parameter - 6 and self.weight_update_final_parameter - 6 <= 9
        assert self.weight_exponent_offset_table_flag in WeightExponentOffsetTableFlag
        assert self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.NOT_INCLUDED or (self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.INCLUDED and self.weight_exponent_offset_flag == WeightExponentOffsetFlag.NOT_ALL_ZERO)
        assert self.weight_exponent_offset_flag == WeightExponentOffsetFlag.ALL_ZERO or self.weight_exponent_offset_table.shape == (self.z_size + 2**16 * (self.z_size == 0), self.prediction_bands_num + int(self.prediction_mode == PredictionMode.FULL))
        if self.weight_exponent_offset_flag == WeightExponentOffsetFlag.NOT_ALL_ZERO:
            for i in range(self.weight_exponent_offset_table.shape[0]):
                for j in range(self.weight_exponent_offset_table.shape[1]):
                    assert -6 <= self.weight_exponent_offset_table[i, j] and self.weight_exponent_offset_table[i, j] <= 5
        assert self.weight_init_method in WeightInitMethod
        assert self.weight_init_table_flag in WeightInitTableFlag
        assert self.weight_init_table_flag == WeightInitTableFlag.NOT_INCLUDED or (self.weight_init_table_flag == WeightInitTableFlag.INCLUDED and self.weight_init_method == WeightInitMethod.CUSTOM)
        assert self.weight_init_method == WeightInitMethod.DEFAULT or self.weight_init_table.shape == (self.z_size + 2**16 * (self.z_size == 0), self.prediction_bands_num + 3 * int(self.prediction_mode == PredictionMode.FULL))
        if self.weight_init_method == WeightInitMethod.CUSTOM:
            for i in range(self.weight_init_table.shape[0]):
                for j in range(self.weight_init_table.shape[1]):
                    assert -2**(self.weight_init_resolution - 1) <= self.weight_init_table[i, j] and self.weight_init_table[i, j] <= 2**(self.weight_init_resolution - 1) - 1
        assert (self.weight_init_method == WeightInitMethod.CUSTOM and 3 <= self.weight_init_resolution and self.weight_init_resolution <= self.weight_component_resolution + 4 + 3) or (self.weight_init_method == WeightInitMethod.DEFAULT and self.weight_init_resolution == 0)

        assert self.periodic_error_updating_flag in PeriodicErrorUpdatingFlag
        assert self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED and self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.LOSSLESS or self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED
        assert (self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED and 0 <= self.error_update_period_exponent and self.error_update_period_exponent <= 9) or (self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED and self.error_update_period_exponent == 0)
        assert self.absolute_error_limit_assignment_method in ErrorLimitAssignmentMethod
        assert 0 < self.get_absolute_error_limit_bit_depth_value() and self.get_absolute_error_limit_bit_depth_value() <= min(self.get_dynamic_range_bits() - 1, 16) or self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.LOSSLESS or self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.RELATIVE_ONLY
        assert (0 <= self.absolute_error_limit_value and self.absolute_error_limit_value <= 2**self.get_absolute_error_limit_bit_depth_value() - 1) or self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED or self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT
        if self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.ABSOLUTE_ONLY or \
            self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.ABSOLUTE_AND_RELATIVE:
            if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED:
                for z in range(self.absolute_error_limit_table.shape[0]):
                    assert 0 <= self.absolute_error_limit_table[z] and self.absolute_error_limit_table[z] <= 2**self.get_absolute_error_limit_bit_depth_value() - 1
            elif self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED:
                for i in range(self.periodic_absolute_error_limit_table.shape[0]):
                    for z in range(self.periodic_absolute_error_limit_table.shape[1]):
                        assert 0 <= self.periodic_absolute_error_limit_table[i, z] and self.periodic_absolute_error_limit_table[i, z] <= 2**self.get_absolute_error_limit_bit_depth_value() - 1
        assert self.relative_error_limit_assignment_method in ErrorLimitAssignmentMethod
        assert 0 < self.get_relative_error_limit_bit_depth_value() and self.get_relative_error_limit_bit_depth_value() <= min(self.get_dynamic_range_bits() - 1, 16) or self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.LOSSLESS or self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.ABSOLUTE_ONLY
        assert (0 <= self.relative_error_limit_value and self.relative_error_limit_value <= 2**self.get_relative_error_limit_bit_depth_value() - 1) or self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED or self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT
        if self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.RELATIVE_ONLY or \
            self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.ABSOLUTE_AND_RELATIVE:
            if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED:
                for z in range(self.relative_error_limit_table.shape[0]):
                    assert 0 <= self.relative_error_limit_table[z] and self.relative_error_limit_table[z] <= 2**self.get_relative_error_limit_bit_depth_value() - 1
            elif self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.USED:
                for i in range(self.periodic_relative_error_limit_table.shape[0]):
                    for z in range(self.periodic_relative_error_limit_table.shape[1]):
                        assert 0 <= self.periodic_relative_error_limit_table[i, z] and self.periodic_relative_error_limit_table[i, z] <= 2**self.get_relative_error_limit_bit_depth_value() - 1

        assert 0 <= self.sample_representative_resolution and self.sample_representative_resolution <= 4
        assert 0 < self.sample_representative_resolution and self.sample_representative_resolution <= 4 and self.sample_representative_flag == SampleRepresentativeFlag.INCLUDED or self.sample_representative_flag == SampleRepresentativeFlag.NOT_INCLUDED
        assert self.band_varying_damping_flag in BandVaryingDampingFlag
        assert self.damping_table_flag in DampingTableFlag
        assert self.damping_table_flag == DampingTableFlag.NOT_INCLUDED or self.damping_table_flag == DampingTableFlag.INCLUDED and self.band_varying_damping_flag == BandVaryingDampingFlag.BAND_DEPENDENT
        assert (self.damping_table_flag == DampingTableFlag.NOT_INCLUDED and 0 <= self.fixed_damping_value and self.fixed_damping_value <= 2**self.sample_representative_resolution - 1) or (self.damping_table_flag == DampingTableFlag.INCLUDED and self.fixed_damping_value == 0)
        assert self.band_varying_offset_flag in BandVaryingOffsetFlag
        assert self.damping_offset_table_flag in OffsetTableFlag
        assert self.damping_offset_table_flag == OffsetTableFlag.NOT_INCLUDED or self.damping_offset_table_flag == OffsetTableFlag.INCLUDED and self.band_varying_offset_flag == BandVaryingOffsetFlag.BAND_DEPENDENT
        assert (self.damping_offset_table_flag == OffsetTableFlag.NOT_INCLUDED and 0 <= self.fixed_offset_value and self.fixed_offset_value <= 2**self.sample_representative_resolution - 1) or (self.damping_offset_table_flag == OffsetTableFlag.INCLUDED and self.fixed_offset_value == 0)
        assert self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.LOSSLESS or self.fixed_offset_value == 0
        if self.band_varying_offset_flag == BandVaryingOffsetFlag.BAND_DEPENDENT:
            for damping in self.damping_table_array:
                assert 0 <= damping and damping <= 2**self.sample_representative_resolution - 1
        if self.band_varying_offset_flag == BandVaryingOffsetFlag.BAND_DEPENDENT:
            for offset in self.damping_offset_table_array:
                assert 0 <= offset and offset <= 2**self.sample_representative_resolution - 1 and self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.LOSSLESS or offset == 0 and self.quantizer_fidelity_control_method == QuantizerFidelityControlMethod.LOSSLESS

        assert (8 <= self.unary_length_limit and self.unary_length_limit < 32) or self.unary_length_limit == 0
        assert max(4, self.initial_count_exponent + 8 * int(self.initial_count_exponent == 0) - 1) <= self.rescaling_counter_size + 4 and self.rescaling_counter_size + 4 <= 11
        assert 0 <= self.initial_count_exponent and self.initial_count_exponent < 8
        assert 0 <= self.accumulator_init_constant and self.accumulator_init_constant <= min(self.get_dynamic_range_bits() - 2, 14) or self.accumulator_init_constant == 15
        assert self.accumulator_init_table_flag == AccumulatorInitTableFlag.NOT_INCLUDED or (self.accumulator_init_table_flag == AccumulatorInitTableFlag.INCLUDED and self.accumulator_init_constant == 15)
        assert self.accumulator_init_table_flag in AccumulatorInitTableFlag

        assert self.block_size in range(4)
        assert self.restricted_code_options_flag in RestrictedCodeOptionsFlag
        assert self.restricted_code_options_flag == RestrictedCodeOptionsFlag.UNRESTRICTED or self.restricted_code_options_flag == RestrictedCodeOptionsFlag.RESTRICTED and self.dynamic_range in range(1,5) and self.large_d_flag == LargeDFlag.SMALL_D
        assert self.reference_sample_interval in range(2**12)
    
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
    
    def __encode_supplementary_information_table_structure(self, index):
        bitstream = bitarray()
        bitstream += bin(self.supplementary_information_tables[index].table_type.value)[2:].zfill(2)
        bitstream += 2 * '0' # Reserved
        bitstream += bin(self.supplementary_information_tables[index].table_purpose)[2:].zfill(4)
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.supplementary_information_tables[index].table_structure.value)[2:].zfill(2)
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.supplementary_information_tables[index].user_defined_data)[2:].zfill(4)
        assert len(bitstream) == 2 * 8
        bitstream += self.supplementary_information_tables[index].table_data_subblock
        assert len(bitstream) % 8 == 0
        return bitstream
    
    def __encode_predictor_primary_structure(self):
        header_bitstream = bitarray()
        optional_tables_bitstream = bitarray()
        
        header_bitstream += 1 * '0' # Reserved
        header_bitstream += bin(self.sample_representative_flag.value)[2:].zfill(1)
        header_bitstream += bin(self.prediction_bands_num)[2:].zfill(4)
        header_bitstream += bin(self.prediction_mode.value)[2:].zfill(1)
        header_bitstream += bin(self.weight_exponent_offset_flag.value)[2:].zfill(1)
        header_bitstream += bin(self.local_sum_type.value)[2:].zfill(2)
        header_bitstream += bin(self.register_size)[2:].zfill(6)
        header_bitstream += bin(self.weight_component_resolution)[2:].zfill(4)
        header_bitstream += bin(self.weight_update_change_interval)[2:].zfill(4)
        header_bitstream += bin(self.weight_update_initial_parameter)[2:].zfill(4)
        header_bitstream += bin(self.weight_update_final_parameter)[2:].zfill(4)
        header_bitstream += bin(self.weight_exponent_offset_table_flag.value)[2:].zfill(1)
        header_bitstream += bin(self.weight_init_method.value)[2:].zfill(1)
        header_bitstream += bin(self.weight_init_table_flag.value)[2:].zfill(1)
        header_bitstream += bin(self.weight_init_resolution)[2:].zfill(5)
        assert len(header_bitstream) == 8 * 5
        
        if self.weight_init_method == WeightInitMethod.CUSTOM:
            for z in range(self.weight_init_table.shape[0]):
                for j in range(min(z, self.prediction_bands_num) + 3 * int(self.prediction_mode == PredictionMode.FULL)):
                    # Transform into two's complement
                    number = self.weight_init_table[z, j]
                    if bin(number)[0] == '-':
                        number += 2**self.weight_init_resolution
                    number = bin(number)[2:].zfill(self.weight_init_resolution)
                    if self.weight_init_table_flag == WeightInitTableFlag.INCLUDED:
                        header_bitstream += number
                    elif self.weight_init_table_flag == WeightInitTableFlag.NOT_INCLUDED:
                        optional_tables_bitstream += number
            if self.weight_init_table_flag == WeightInitTableFlag.INCLUDED:
                fill_bits = (8 - len(header_bitstream) % 8) % 8
                header_bitstream += fill_bits * '0'
                assert len(header_bitstream) % 8 == 0
            elif self.weight_init_table_flag == WeightInitTableFlag.NOT_INCLUDED:
                fill_bits = (8 - len(optional_tables_bitstream) % 8) % 8
                optional_tables_bitstream += fill_bits * '0'
                assert len(optional_tables_bitstream) % 8 == 0
            
        if self.weight_exponent_offset_flag == WeightExponentOffsetFlag.NOT_ALL_ZERO:
            for z in range(self.weight_exponent_offset_table.shape[0]):
                for j in range(min(z, self.prediction_bands_num) + int(self.prediction_mode == PredictionMode.FULL)):
                    # Transform into two's complement
                    number = self.weight_exponent_offset_table[z, j]
                    if bin(number)[0] == '-':
                        number += 2**4
                    number = bin(number)[2:].zfill(4)
                    if self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.INCLUDED:
                        header_bitstream += number
                    elif self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.NOT_INCLUDED:
                        optional_tables_bitstream += number
            if self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.INCLUDED:
                fill_bits = (8 - len(header_bitstream) % 8) % 8
                header_bitstream += fill_bits * '0'
                assert len(header_bitstream) % 8 == 0
            elif self.weight_exponent_offset_table_flag == WeightExponentOffsetTableFlag.NOT_INCLUDED:
                fill_bits = (8 - len(optional_tables_bitstream) % 8) % 8
                optional_tables_bitstream += fill_bits * '0'
                assert len(optional_tables_bitstream) % 8 == 0
        
        return header_bitstream, optional_tables_bitstream
    
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
        if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED:
            error_limit_bit_depth = self.get_absolute_error_limit_bit_depth_value()
            if self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                bitstream += bin(self.absolute_error_limit_value)[2:].zfill(error_limit_bit_depth)
                assert len(bitstream) == error_limit_bit_depth + 8
            elif self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                for z in range(self.absolute_error_limit_table.shape[0]):
                    bitstream += bin(self.absolute_error_limit_table[z])[2:].zfill(error_limit_bit_depth)
                assert len(bitstream) == self.absolute_error_limit_table.shape[0] * error_limit_bit_depth + 8
            fill_bits = (8 - len(bitstream) % 8) % 8
            bitstream += fill_bits * '0'
        assert len(bitstream) % 8 == 0
        return bitstream
    
    def __encode_predictor_quantization_relative_error_limit_structure(self):
        bitstream = bitarray()
        bitstream += 1 * '0' # Reserved
        bitstream += bin(self.relative_error_limit_assignment_method.value)[2:].zfill(1)
        bitstream += 2 * '0' # Reserved
        bitstream += bin(self.relative_error_limit_bit_depth)[2:].zfill(4)
        if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED:
            error_limit_bit_depth = self.get_relative_error_limit_bit_depth_value()
            if self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                bitstream += bin(self.relative_error_limit_value)[2:].zfill(error_limit_bit_depth)
                assert len(bitstream) == error_limit_bit_depth + 8
            elif self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                for z in range(self.relative_error_limit_table.shape[0]):
                    bitstream += bin(self.relative_error_limit_table[z])[2:].zfill(error_limit_bit_depth)    
                assert len(bitstream) == self.relative_error_limit_table.shape[0] * error_limit_bit_depth + 8
            fill_bits = (8 - len(bitstream) % 8) % 8
            bitstream += fill_bits * '0'
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
        header_bitstream = bitarray()
        optional_tables_bitstream = bitarray()

        header_bitstream += 5 * '0' # Reserved
        header_bitstream += bin(self.sample_representative_resolution)[2:].zfill(3)
        header_bitstream += 1 * '0' # Reserved
        header_bitstream += bin(self.band_varying_damping_flag.value)[2:].zfill(1)
        header_bitstream += bin(self.damping_table_flag.value)[2:].zfill(1)
        header_bitstream += 1 * '0' # Reserved
        header_bitstream += bin(self.fixed_damping_value)[2:].zfill(4)
        header_bitstream += 1 * '0' # Reserved
        header_bitstream += bin(self.band_varying_offset_flag.value)[2:].zfill(1)
        header_bitstream += bin(self.damping_offset_table_flag.value)[2:].zfill(1)
        header_bitstream += 1 * '0'
        header_bitstream += bin(self.fixed_offset_value)[2:].zfill(4)
        assert len(header_bitstream) == 8 * 3
        
        bitstream = bitarray()
        if self.band_varying_damping_flag == BandVaryingDampingFlag.BAND_DEPENDENT:
            for z in range(self.damping_table_array.shape[0]):
                bitstream += bin(self.damping_table_array[z])[2:].zfill(self.sample_representative_resolution)
            fill_bits = (8 - len(bitstream) % 8) % 8
            bitstream += fill_bits * '0'
            assert len(bitstream) % 8 == 0
        if self.damping_table_flag == DampingTableFlag.INCLUDED:
            header_bitstream += bitstream
        elif self.damping_table_flag == DampingTableFlag.NOT_INCLUDED:
            optional_tables_bitstream += bitstream
        
        bitstream = bitarray()
        if self.band_varying_offset_flag == BandVaryingOffsetFlag.BAND_DEPENDENT:
            for z in range(self.damping_offset_table_array.shape[0]):
                bitstream += bin(self.damping_offset_table_array[z])[2:].zfill(self.sample_representative_resolution)
            fill_bits = (8 - len(bitstream) % 8) % 8
            bitstream += fill_bits * '0'
            assert len(bitstream) % 8 == 0
        if self.damping_offset_table_flag == OffsetTableFlag.INCLUDED:
            header_bitstream += bitstream
        elif self.damping_offset_table_flag == OffsetTableFlag.NOT_INCLUDED:
            optional_tables_bitstream += bitstream
                
        return header_bitstream, optional_tables_bitstream

    def __encode_entropy_coder_sample_adaptive_structure(self):
        header_bitstream = bitarray()
        optional_tables_bitstream = bitarray()
        
        header_bitstream += bin(self.unary_length_limit)[2:].zfill(5)
        header_bitstream += bin(self.rescaling_counter_size)[2:].zfill(3)
        header_bitstream += bin(self.initial_count_exponent)[2:].zfill(3)
        header_bitstream += bin(self.accumulator_init_constant)[2:].zfill(4)
        header_bitstream += bin(self.accumulator_init_table_flag.value)[2:].zfill(1)
        assert len(header_bitstream) == 8 * 2

        bitstream = bitarray()
        if self.accumulator_init_constant == 15:
            for z in range(self.accumulator_init_table.shape[0]):
                bitstream += bin(self.accumulator_init_table[z])[2:].zfill(4)
            fill_bits = (8 - len(bitstream) % 8) % 8
            bitstream += fill_bits * '0'
            assert len(bitstream) % 8 == 0
        
        if self.accumulator_init_table_flag == AccumulatorInitTableFlag.INCLUDED:
            header_bitstream += bitstream
        elif self.accumulator_init_table_flag == AccumulatorInitTableFlag.NOT_INCLUDED:
            optional_tables_bitstream += bitstream
   
        return header_bitstream, optional_tables_bitstream
    
    def __encode_entropy_coder_hybrid_structure(self):
        bitstream = bitarray()
        bitstream += bin(self.unary_length_limit)[2:].zfill(5)
        bitstream += bin(self.rescaling_counter_size)[2:].zfill(3)
        bitstream += bin(self.initial_count_exponent)[2:].zfill(3)
        bitstream += 5 * '0' # Reserved
        assert len(bitstream) == 8 * 2
        return bitstream
    
    def __encode_entropy_coder_block_adaptive_structure(self):
        bitstream = bitarray()
        bitstream += 1 * '0'
        bitstream += bin(self.block_size)[2:].zfill(2)
        bitstream += bin(self.restricted_code_options_flag.value)[2:].zfill(1)
        bitstream += bin(self.reference_sample_interval)[2:].zfill(12)
        assert len(bitstream) == 8 * 2
        return bitstream
    
    def __create_header_bitstream(self):
        header_bitstream = bitarray()
        optional_tables_bitstream = bitarray()

        header_bitstream += self.__encode_essential_subpart_structure()

        for index in range(self.supplementary_information_table_count):
            header_bitstream += self.__encode_supplementary_information_table_structure(index)

        bitstreams = self.__encode_predictor_primary_structure()
        header_bitstream += bitstreams[0]
        optional_tables_bitstream += bitstreams[1]

        if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.LOSSLESS:
            header_bitstream += self.__encode_predictor_quantization_structure()
        if self.sample_representative_flag == SampleRepresentativeFlag.INCLUDED:
            bitstreams = self.__encode_predictor_sample_representative_structure()
            header_bitstream += bitstreams[0]
            optional_tables_bitstream += bitstreams[1]
        if self.entropy_coder_type == EntropyCoderType.SAMPLE_ADAPTIVE:
            bitstreams =  self.__encode_entropy_coder_sample_adaptive_structure()
            header_bitstream += bitstreams[0]
            optional_tables_bitstream += bitstreams[1]
        elif self.entropy_coder_type == EntropyCoderType.HYBRID:
            header_bitstream += self.__encode_entropy_coder_hybrid_structure()
        elif self.entropy_coder_type == EntropyCoderType.BLOCK_ADAPTIVE:
            header_bitstream += self.__encode_entropy_coder_block_adaptive_structure()
        
        self.header_bitstream = header_bitstream
        self.optional_tables_bitstream = optional_tables_bitstream

    def set_encoding_order_bip(self):
        self.sample_encoding_order = SampleEncodingOrder.BI
        self.sub_frame_interleaving_depth = self.z_size

    def set_encoding_order_bil(self):
        self.sample_encoding_order = SampleEncodingOrder.BI
        self.sub_frame_interleaving_depth = 1

    def set_weight_init_table_array_to_default(self):
        # The default values here are arbitrary, not set from standard
        self.__init_weight_init_table_array()
        for z in range(self.weight_init_table.shape[0]):
            for j in range(min(z, self.prediction_bands_num) + 3 * int(self.prediction_mode == PredictionMode.FULL)):
                self.weight_init_table[z, j] = self.weight_init_table_value
    
    def set_weight_exponent_offset_table_array_to_default(self):
        # The default values here are arbitrary, not set from standard
        self.__init_weight_exponent_offset_table_array()
        for z in range(self.weight_exponent_offset_table.shape[0]):
            for j in range(min(z, self.prediction_bands_num) + int(self.prediction_mode == PredictionMode.FULL)):
                self.weight_exponent_offset_table[z, j] = self.weight_exponent_offset_value
    
    def set_absolute_error_limit_table_array_to_default(self):
        # The default values here are arbitrary, not set from standard
        self.__init_absolute_error_limit_table_array()
        for z in range(self.absolute_error_limit_table.shape[0]):
                if self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                    self.absolute_error_limit_table[z] = self.absolute_error_limit_value
                elif self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                    self.absolute_error_limit_table[z] = min(z, 2**self.get_absolute_error_limit_bit_depth_value() - 1)

    def set_relative_error_limit_table_array_to_default(self):
        # The default values here are arbitrary, not set from standard
        self.__init_relative_error_limit_table_array()
        for z in range(self.relative_error_limit_table.shape[0]):
                if self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                    self.relative_error_limit_table[z] = self.relative_error_limit_value
                elif self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                    self.relative_error_limit_table[z] = min(self.relative_error_limit_value + 2 * z, 2**self.get_relative_error_limit_bit_depth_value() - 1)
    
    def set_periodic_absolute_error_limit_table_array_to_default(self):
        # The default values here are arbitrary, not set from standard
        self.__init_periodic_absolute_error_limit_table_array()
        for i in range(self.periodic_absolute_error_limit_table.shape[0]):
            for z in range(self.periodic_absolute_error_limit_table.shape[1]):
                if self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                    self.periodic_absolute_error_limit_table[i,z] = (i * self.periodic_absolute_error_limit_table.shape[1]) % (2**self.get_absolute_error_limit_bit_depth_value() - 1)
                elif self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                    self.periodic_absolute_error_limit_table[i,z] = (i * self.periodic_absolute_error_limit_table.shape[1] + z) % (2**self.get_absolute_error_limit_bit_depth_value() - 1)
    
    def set_periodic_relative_error_limit_table_array_to_default(self):
        # The default values here are arbitrary, not set from standard
        self.__init_periodic_relative_error_limit_table_array()
        for i in range(self.periodic_relative_error_limit_table.shape[0]):
            for z in range(self.periodic_relative_error_limit_table.shape[1]):
                if self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                    self.periodic_relative_error_limit_table[i,z] = (i * self.periodic_relative_error_limit_table.shape[1]) % (2**self.get_relative_error_limit_bit_depth_value() - 1)
                elif self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                    self.periodic_relative_error_limit_table[i,z] = (i * self.periodic_relative_error_limit_table.shape[1] + z) % (2**self.get_relative_error_limit_bit_depth_value() - 1)

    def set_damping_table_array_to_default(self):
        # The default values here are arbitrary, not set from standard
        self.__init_damping_table_array()
        if self.band_varying_damping_flag == BandVaryingDampingFlag.BAND_INDEPENDENT:
            self.damping_table_array[:] = self.fixed_damping_value
        elif self.band_varying_damping_flag == BandVaryingDampingFlag.BAND_DEPENDENT:
            for z in range(self.damping_table_array.shape[0]):
                self.damping_table_array[z] = z % (2**self.sample_representative_resolution - 1)
    
    def set_damping_offset_table_array_to_default(self):
        # The default values here are arbitrary, not set from standard
        self.__init_damping_offset_table_array()
        if self.band_varying_offset_flag == BandVaryingOffsetFlag.BAND_INDEPENDENT:
            self.damping_offset_table_array[:] = self.fixed_offset_value
        elif self.band_varying_offset_flag == BandVaryingOffsetFlag.BAND_DEPENDENT:
            for z in range(self.damping_offset_table_array.shape[0]):
                self.damping_offset_table_array[z] = z % (2**self.sample_representative_resolution - 1)
    
    def set_accumulator_init_table_to_default(self):
        # The default values here are arbitrary, not set from standard
        self.__init_accumulator_init_table()
        if self.accumulator_init_constant == 15:
            for z in range(self.accumulator_init_table.shape[0]):
                self.accumulator_init_table[z] = z % min(self.get_dynamic_range_bits() - 2, 14)
        else:
            self.accumulator_init_table[:] = self.accumulator_init_constant
      
    def get_dynamic_range_bits(self):
        dynamic_range_bits = self.dynamic_range
        if dynamic_range_bits == 0:
            dynamic_range_bits = 16
        if self.large_d_flag == LargeDFlag.LARGE_D:
            dynamic_range_bits += 16
        return dynamic_range_bits
    
    def get_absolute_error_limit_bit_depth_value(self):
        return self.absolute_error_limit_bit_depth + 16 * int(self.absolute_error_limit_bit_depth == 0)

    def get_relative_error_limit_bit_depth_value(self):
        return self.relative_error_limit_bit_depth + 16 * int(self.relative_error_limit_bit_depth == 0)
    
    def get_header_bitstreams(self):
        self.__create_header_bitstream()
        return self.header_bitstream, self.optional_tables_bitstream
    
    def get_error_limits_bitstream(self):
        bitstream = bitarray()
        if self.periodic_error_updating_flag == PeriodicErrorUpdatingFlag.NOT_USED:
            return bitstream

        for i in range(ceil((self.y_size + 2**16 * int(self.y_size == 0)) / 2**self.error_update_period_exponent)):

            if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.RELATIVE_ONLY:
                if self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                    bitstream += bin(self.periodic_absolute_error_limit_table[i,0])[2:].zfill(16)
                elif self.absolute_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                    for z in range(self.periodic_absolute_error_limit_table.shape[1]):
                        bitstream += bin(self.periodic_absolute_error_limit_table[i,z])[2:].zfill(16)

            if self.quantizer_fidelity_control_method != QuantizerFidelityControlMethod.ABSOLUTE_ONLY:
                if self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_INDEPENDENT:
                    bitstream += bin(self.periodic_relative_error_limit_table[i,0])[2:].zfill(16)
                elif self.relative_error_limit_assignment_method == ErrorLimitAssignmentMethod.BAND_DEPENDENT:
                    for z in range(self.periodic_relative_error_limit_table.shape[1]):
                        bitstream += bin(self.periodic_relative_error_limit_table[i,z])[2:].zfill(16)
        
        return bitstream

    def save(self, output_folder):
        bitstreams = self.get_header_bitstreams()
        with open(output_folder + "/header.bin", "wb") as file:
            bitstreams[0].tofile(file)
        with open(output_folder + "/optional_tables.bin", "wb") as file:
            bitstreams[1].tofile(file)
        
        with open(output_folder + "/error_limits.bin", "wb") as file:
            self.get_error_limits_bitstream().tofile(file)
        
        if type(self.periodic_absolute_error_limit_table) is np.ndarray:
            np.savetxt(output_folder + "/header-00-periodic_absolute_error_limit_table.csv", self.periodic_absolute_error_limit_table, delimiter=",", fmt='%d')
        if type(self.periodic_relative_error_limit_table) is np.ndarray:
            np.savetxt(output_folder + "/header-01-periodic_relative_error_limit_table.csv", self.periodic_relative_error_limit_table, delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/header-02-damping_table_array.csv", self.damping_table_array, delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/header-03-damping_offset_table_array.csv", self.damping_offset_table_array, delimiter=",", fmt='%d')
        np.savetxt(output_folder + "/header-04-accumulator_init_table.csv", self.accumulator_init_table, delimiter=",", fmt='%d')
    