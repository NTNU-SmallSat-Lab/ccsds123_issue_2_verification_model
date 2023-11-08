from . import header as hd
from . import constants as const
from . import predictor as pred
from . import sa_encoder as sa_enc
from . import hybrid_encoder as hyb_enc
from . import ba_encoder as ba_enc
import numpy as np
import re
import time

class CCSDS123():
    """
    CCSDS 123.0-B-2 high level model class
    """
    header = None
    predictor = None
    image_file = None
    image_name = None
    image_sample = None # Symbol: s
    output_folder = "output"
    header_file = None
    use_header_file = False
    accu_init_file = None
    use_accu_init_file = False

    def __init__(self, image_file):
        self.image_file = image_file
        self.image_name = image_file.split('/')[-1]
        print(self.image_name)

    def __get_sample_format(self):
        formats = {
            "u8be": np.dtype('>u1'), "u8le": np.dtype('<u1'), "s8be": np.dtype('>i1'), "s8le": np.dtype('<i1'),
            "u16be": np.dtype('>u2'), "u16le": np.dtype('<u2'), "s16be": np.dtype('>i2'),"s16le": np.dtype('<i2'),
            "u32be": np.dtype('>u4'), "u32le": np.dtype('<u4'), "s32be": np.dtype('>i4'), "s32le": np.dtype('<i4'),
            "u64be": np.dtype('>u8'), "u64le": np.dtype('<u8'), "s64be": np.dtype('>i8'), "s64le": np.dtype('<i8'),
        }
        format = self.image_name.split('-')[-2].split('-')[-1]
        return formats[format]

    def __load_raw_image(self):
        """Load a raw image into a N_x * N_y by N_z array"""

        if "le" == re.findall('-(.*)-', self.image_file)[0][-2:]:
            print("Little endian is not tested")
        self.image_sample = np.fromfile(self.image_file, dtype=self.__get_sample_format())
        self.image_sample = self.image_sample.astype(dtype=np.int64)
        self.image_sample = self.image_sample.reshape((self.header.z_size, self.header.y_size, self.header.x_size)) # Reshape to z,y,x (BSQ) 3D array
        self.image_sample = self.image_sample.transpose(1,2,0) # Transpose to y,x,z order (BIP) 

    def set_header_file(self, header_file):
        self.header_file = header_file
        self.use_header_file = True

    def set_hybrid_accu_init_file(self, accu_init_file):
        self.accu_init_file = accu_init_file
        self.use_accu_init_file = True

    def compress_image(self):
        start_time = time.time()

        self.header = hd.Header(self.image_file)
        if self.use_header_file:
            self.header.set_config_from_file(self.header_file)

        self.__load_raw_image()
        print(f"{time.time() - start_time:.3f} seconds. Done with loading")

        self.image_constants = const.ImageConstants(self.header)

        self.predictor = pred.Predictor(self.header, self.image_constants, self.image_sample)
        self.predictor.run_predictor()
        print(f"{time.time() - start_time:.3f} seconds. Done with predictor")

        if self.header.entropy_coder_type == hd.EntropyCoderType.SAMPLE_ADAPTIVE:
            self.encoder = sa_enc.SampleAdaptiveEncoder(self.header,
                                                        self.image_constants,
                                                        self.predictor.get_predictor_output())
        elif self.header.entropy_coder_type == hd.EntropyCoderType.HYBRID:
            self.encoder = hyb_enc.HybridEncoder(self.header,
                                                 self.image_constants,
                                                 self.predictor.get_predictor_output())
            if self.use_accu_init_file:
                self.encoder.set_hybrid_accu_init_file(self.accu_init_file)
        elif self.header.entropy_coder_type == hd.EntropyCoderType.BLOCK_ADAPTIVE:
            self.encoder = ba_enc.BlockAdaptiveEncoder(self.header,
                                                       self.image_constants,
                                                       self.predictor.get_predictor_output())
        
        self.encoder.run_encoder()
        print(f"{time.time() - start_time:.3f} seconds. Done with encoder")

        self.header.save_header(self.output_folder)
        self.predictor.save_data(self.output_folder)
        self.encoder.save_data(self.output_folder, self.header.get_header_bitstream())
        print(f"{time.time() - start_time:.3f} seconds. Done with saving")
        



