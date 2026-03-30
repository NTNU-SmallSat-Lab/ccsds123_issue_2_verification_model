from . import header as hd
from . import constants as const
from . import sa_encoder as sa_enc
from . import hybrid_encoder as hyb_enc
from . import ba_encoder as ba_enc
from . import predictor_old as pred_old
from . import _predictor as pred

import numpy as np
import time
from pathlib import Path


class CCSDS123:
    """
    CCSDS 123.0-B-2 high level model class
    """

    delayed_weight_updates = None
    save_intermediates = None
    use_old_predictor = None

    header = None
    predictor = None
    image_file = None
    image_name = None
    image_ordering = None
    sample_format = None
    image_sample = None  # Symbol: s
    output_folder = str(Path(__file__).resolve().parent.parent) + "/output"
    header_file = None
    optional_tables_file = None
    error_limits_file = None
    use_header_file = False
    accu_init_file = None
    use_accu_init_file = False
    mqi = None

    def __init__(self, image_file, image_ordering="BSQ", delayed_weight_updates=True, save_intermediates=False, use_old_predictor=False):
        self.image_file = image_file
        self.image_name = image_file.split("/")[-1]
        self.image_ordering = image_ordering
        self.delayed_weight_updates = delayed_weight_updates
        self.save_intermediates = save_intermediates
        self.use_old_predictor = use_old_predictor

        if self.delayed_weight_updates:
            print("Using delayed weight updates")

    def get_sample_format(self):
        formats = {
            "u8be": np.dtype(">u1"),
            "u8le": np.dtype("<u1"),
            "s8be": np.dtype(">i1"),
            "s8le": np.dtype("<i1"),
            "u16be": np.dtype(">u2"),
            "u16le": np.dtype("<u2"),
            "s16be": np.dtype(">i2"),
            "s16le": np.dtype("<i2"),
            "u32be": np.dtype(">u4"),
            "u32le": np.dtype("<u4"),
            "s32be": np.dtype(">i4"),
            "s32le": np.dtype("<i4"),
            "u64be": np.dtype(">u8"),
            "u64le": np.dtype("<u8"),
            "s64be": np.dtype(">i8"),
            "s64le": np.dtype("<i8"),
        }
        format = self.image_name.split("-")[-2].split("-")[-1]
        return formats[format]

    def __load_raw_image(self):
        """Load a raw image into a N_x * N_y by N_z array"""
        # This should be updated to support different file formats for the input image

        self.sample_format = self.get_sample_format()

        # Get image from file and convert data type to int64
        self.image_sample = np.fromfile(self.image_file, dtype=self.sample_format)
        self.image_sample = self.image_sample.astype(dtype=np.int64)

        if self.image_ordering == "BSQ":
            self.image_sample = self.image_sample.reshape((self.header.z_size, self.header.y_size, self.header.x_size))  # Reshape to z,y,x (BSQ) 3D array

            self.image_sample = self.image_sample.transpose(1, 2, 0)  # Transpose to y,x,z order (BIP)
        elif self.image_ordering == "BIP":
            self.image_sample = self.image_sample.reshape((self.header.y_size, self.header.x_size, self.header.z_size))  # Image was stored as BIP
        else:
            print(f"Image file ordering {self.image_ordering} is unsupported. Suppurted values are 'BSQ' and 'BIP'.")
            raise RuntimeError

    def set_header_file(self, header_file):
        self.header_file = header_file
        self.use_header_file = True

    def set_optional_tables_file(self, optional_tables_file):
        self.optional_tables_file = optional_tables_file

    def set_error_limits_file(self, error_limits_file):
        self.error_limits_file = error_limits_file

    def set_hybrid_accu_init_file(self, accu_init_file):
        self.accu_init_file = accu_init_file
        self.use_accu_init_file = True

    def set_header(self):
        self.header = hd.Header(self.image_name)
        if self.use_header_file:
            self.header.set_config_from_file(self.header_file, self.optional_tables_file, self.error_limits_file)

    def set_output_dir(self, output):
        self.output_folder = output

    def compress_image(self):
        start_time = time.time()

        self.header = hd.Header(self.image_name)
        if self.use_header_file:
            self.header.set_config_from_file(self.header_file, self.optional_tables_file, self.error_limits_file)

        self.__load_raw_image()
        print(f"{time.time() - start_time:.3f} seconds. Done with loading")

        self.image_constants = const.ImageConstants(self.header)

        if self.use_old_predictor:
            self.predictor_old = pred_old.Predictor(self.header, self.image_constants, self.image_sample, self.delayed_weight_updates)
        else:
            self.predictor = pred.Predictor(self.header, self.image_constants, self.delayed_weight_updates, self.save_intermediates)

        predictor_output = None

        if self.use_old_predictor:
            print("Compressing with old predictor")
            predictor_output = self.predictor_old.run_predictor()
        else:
            print("Compressing with new predictor")
            try:
                predictor_output = self.predictor.compress(self.image_sample)
            except Exception as e:
                self.predictor.save_data(self.output_folder)
                print("Predictor threw exception (", e, "), saving and exiting")
                raise RuntimeError

        self.mqi = predictor_output  # for debugging of decompression

        print(f"{time.time() - start_time:.3f} seconds. Done with predictor")

        if self.header.entropy_coder_type == hd.EntropyCoderType.SAMPLE_ADAPTIVE:
            self.encoder = sa_enc.SampleAdaptiveEncoder(self.header, self.image_constants, predictor_output)
        elif self.header.entropy_coder_type == hd.EntropyCoderType.HYBRID:
            self.encoder = hyb_enc.HybridEncoder(self.header, self.image_constants, predictor_output)
            if self.use_accu_init_file:
                self.encoder.set_hybrid_accu_init_file(self.accu_init_file)
        elif self.header.entropy_coder_type == hd.EntropyCoderType.BLOCK_ADAPTIVE:
            self.encoder = ba_enc.BlockAdaptiveEncoder(self.header, self.image_constants, predictor_output)

        self.encoder.run_encoder()
        print(f"{time.time() - start_time:.3f} seconds. Done with encoder")

        if self.use_old_predictor:
            self.predictor_old.save_data("reference")
        else:
            self.predictor.save_data(self.output_folder)

        self.header.save_data(self.output_folder)
        self.encoder.save_data(self.output_folder, self.header.get_header_bitstreams()[0])

        print(f"{time.time() - start_time:.3f} seconds. Done with saving")

    def decompress_image(self):
        start_time = time.time()

        # need to add support for reading config from compressed bitstream
        self.header = hd.Header(self.image_name)
        if self.use_header_file:
            self.header.set_config_from_file(self.header_file, self.optional_tables_file, self.error_limits_file)

        self.image_constants = const.ImageConstants(self.header)

        print(f"{time.time() - start_time:.3f} seconds. Done with loading")

        print("Decompressing")

        self.predictor = pred.Predictor(self.header, self.image_constants, self.delayed_weight_updates, self.save_intermediates)

        decompressed = None
        try:
            decompressed = self.predictor.decompress(self.mqi)
        except Exception as e:
            raise e

        print(f"{time.time() - start_time:.3f} seconds. Done with predictor")

        csv_image_shape = (self.header.y_size * self.header.x_size, self.header.z_size)
        np.savetxt(
            f"{self.output_folder}/predictor-16-image_sample.csv",
            decompressed.reshape(csv_image_shape),
            delimiter=",",
            fmt="%d",
        )

        # already is in BIP, (y,x,z)
        if self.header.sample_encoding_order == hd.SampleEncodingOrder.BSQ:
            decompressed = decompressed.transpose(2, 0, 1)  # (z,y,x)
        else:
            if self.header.sub_frame_interleaving_depth == 1:  # BIL
                decompressed = decompressed.transpose(0, 2, 1)  # (y,z,x)
            elif self.header.sub_frame_interleaving_depth == self.header.z_size:  # BIP
                pass
            else:
                print("BRUH!")  # TODO: fix

        self.predictor.save_data(self.output_folder)
        with open(f"{self.output_folder}/z-output-bitstream-dec.bin", "wb") as f:
            f.write(decompressed.astype(self.get_sample_format()).tobytes())

        print(f"{time.time() - start_time:.3f} seconds. Done with saving")
