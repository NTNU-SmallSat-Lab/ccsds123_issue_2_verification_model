from . import header as hd


class ImageConstants():
    """Class to hold constants across the compression process"""
    header = None
    
    def __init__(self, header):
        self.header = header
        self.__init_image_constants()

    # Image constants
    dynamic_range_bits = None # Symbol: D
    dynamic_range = None # 2^D
    lower_sample_limit = None # Symbol: s_min
    upper_sample_limit = None # Symbol: s_max
    middle_sample_value = None # Symbol: s_mid

    def __init_image_constants(self):
        self.dynamic_range_bits = self.header.dynamic_range
        if self.dynamic_range_bits == 0:
            self.dynamic_range_bits = 16
        if self.header.large_d_flag == hd.LargeDFlag.LARGE_D:
            self.dynamic_range_bits += 16

        self.dynamic_range = 2**self.dynamic_range_bits

        if self.header.sample_type == hd.SampleType.UNSIGNED_INTEGER:
            self.lower_sample_limit = 0
            self.upper_sample_limit = 2 ** self.dynamic_range_bits - 1
            self.middle_sample_value = 2 ** (self.dynamic_range_bits - 1)
        elif self.header.sample_type == hd.SampleType.SIGNED_INTEGER:
            self.lower_sample_limit = -2 ** (self.dynamic_range_bits - 1)
            self.upper_sample_limit = 2 ** (self.dynamic_range_bits - 1) - 1
            self.middle_sample_value = 0
