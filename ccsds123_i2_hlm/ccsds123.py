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
from bitarray import bitarray


class CCSDS123:
    """
    CCSDS 123.0-B-2 high level model class
    """

    delayed_weight_updates = None
    save_intermediates = None
    use_old_predictor = None
    predictor_only = None

    header = None
    predictor = None
    image_file = None
    image_name = None
    compressed_image_file = None
    compressed_body_size = None
    image_ordering = None
    sample_format = None
    image_sample = None  # Symbol: s
    compressed_bitstream = None
    output_folder = str(Path(__file__).resolve().parent.parent) + "/output"
    header_file = None
    optional_tables_file = None
    error_limits_file = None
    use_header_file = False
    accu_init_file = None
    use_accu_init_file = False
    mapped_quantizer_index = None

    def __init__(self, image_ordering="BSQ", delayed_weight_updates=False, save_intermediates=False, use_old_predictor=False, predictor_only=False):
        self.image_ordering = image_ordering
        self.delayed_weight_updates = delayed_weight_updates
        self.save_intermediates = save_intermediates
        self.use_old_predictor = use_old_predictor
        self.predictor_only = predictor_only

        if self.delayed_weight_updates:
            print("Using delayed weight updates")

    def get_sample_format(self, format=None):
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
        if format == None:
            if self.image_name == None:
                raise RuntimeError("File format not provided and cannot get file format from file_name")
            format = self.image_name.split("-")[-2].split("-")[-1]
        return formats[format]

    def __load_raw_image(self, file_format=None):
        """Load a raw image into a N_x * N_y by N_z array"""
        # This should be updated to support different file formats for the input image

        self.sample_format = self.get_sample_format(file_format)

        # Get image from file and convert data type to int64
        self.image_sample = np.fromfile(self.image_file, dtype=self.sample_format)
        self.image_sample = self.image_sample.astype(dtype=np.int64)

        if self.image_ordering == "BSQ":
            self.image_sample = self.image_sample.reshape((self.header.z_size, self.header.y_size, self.header.x_size))  # Reshape to z,y,x (BSQ) 3D array
            self.image_sample = self.image_sample.transpose(1, 2, 0)  # Transpose to y,x,z order (BIP)
        elif self.image_ordering == "BIL":
            self.image_sample = self.image_sample.reshape((self.header.y_size, self.header.z_size, self.header.x_size))  # Reshape to y,z,x (BIL) 3D array
            self.image_sample = self.image_sample.transpose(0, 2, 1)  # Transpose to y,x,z order (BIP)
        elif self.image_ordering == "BIP":
            self.image_sample = self.image_sample.reshape((self.header.y_size, self.header.x_size, self.header.z_size))  # Image was stored as BIP
        else:
            print(f"Image file ordering {self.image_ordering} is unsupported. Suppurted values are 'BSQ', 'BIL' and 'BIP'.")
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
        self.header = hd.Header(self.image_name, self.save_intermediates)
        if self.use_header_file:
            self.header.set_config_from_file(self.header_file, self.optional_tables_file, self.error_limits_file)

    def set_output_dir(self, output):
        self.output_folder = output

    def compress_image(self, image_file, file_format=None):
        print(f"-I- Compressing '{image_file}'")

        start_time = time.time()

        self.image_file = image_file
        self.image_name = image_file.split("/")[-1]

        self.header = hd.Header(self.image_name, self.save_intermediates)
        if self.use_header_file:
            self.header.set_config_from_file(self.header_file, self.optional_tables_file, self.error_limits_file)

        self.__load_raw_image(file_format)
        print(f"{time.time() - start_time:.3f} seconds. Done with loading")

        self.image_constants = const.ImageConstants(self.header)

        if self.use_old_predictor:
            self.predictor_old = pred_old.Predictor(self.header, self.image_constants, self.image_sample, self.delayed_weight_updates)
        else:
            self.predictor = pred.Predictor(self.header, self.image_constants, self.delayed_weight_updates, self.save_intermediates)

        if self.use_old_predictor:
            print("-W- Using old predictor")
            self.mapped_quantizer_index = self.predictor_old.run_predictor()
        else:
            try:
                self.mapped_quantizer_index = self.predictor.compress(self.image_sample)
            except Exception as e:
                self.predictor.save_data(self.output_folder)
                print("Predictor threw exception (", e, "), saving and exiting")
                raise RuntimeError

        print(f"{time.time() - start_time:.3f} seconds. Done with predictor")

        if self.header.entropy_coder_type == hd.EntropyCoderType.SAMPLE_ADAPTIVE:
            self.encoder = sa_enc.SampleAdaptiveEncoder(self.header, self.image_constants)
        elif self.header.entropy_coder_type == hd.EntropyCoderType.HYBRID:
            self.encoder = hyb_enc.HybridEncoder(self.header, self.image_constants, self.save_intermediates)
            if self.use_accu_init_file:
                self.encoder.set_hybrid_accu_init_file(self.accu_init_file)
        elif self.header.entropy_coder_type == hd.EntropyCoderType.BLOCK_ADAPTIVE:
            self.encoder = ba_enc.BlockAdaptiveEncoder(self.header, self.image_constants)

        self.encoder.run_encoder(self.mapped_quantizer_index)
        print(f"{time.time() - start_time:.3f} seconds. Done with encoder")

        if self.use_old_predictor:
            self.predictor_old.save_data(self.output_folder)
        else:
            self.predictor.save_data(self.output_folder)

        self.header.save_data(self.output_folder)
        self.encoder.save_data(self.output_folder, self.header.get_header_bitstreams()[0])

        print(f"{time.time() - start_time:.3f} seconds. Done with saving")

    def decompress_image(self, compressed_image_file, output_format=None):
        print(f"-I- Decompressing '{compressed_image_file}'")

        start_time = time.time()

        self.compressed_image_file = compressed_image_file

        ########################################################################################################################
        # Reading config
        ########################################################################################################################

        if not self.predictor_only:  # header set from previous run of compressor
            self.compressed_bitstream = bitarray()
            with open(self.compressed_image_file, "rb") as file:
                self.compressed_bitstream.fromfile(file)  # we assume here that the compressed bitstream is in big endian

            print("Reading header config from compressed bitstream")
            self.header = hd.Header(save_intermediates=self.save_intermediates)  # by not passing image_name we read image size from header bitstream instead
            self.compressed_body_size = self.header.set_config_from_file(self.compressed_bitstream, self.optional_tables_file)
            self.image_constants = const.ImageConstants(self.header)

        print(f"{time.time() - start_time:.3f} seconds. Done with loading")

        ########################################################################################################################
        # Decompression
        ########################################################################################################################

        if not self.predictor_only:  # assumes mapped_quantizer_index is available from previous run of compressor
            if self.header.entropy_coder_type == hd.EntropyCoderType.HYBRID:
                self.encoder = hyb_enc.HybridEncoder(self.header, self.image_constants, self.save_intermediates)
                if self.use_accu_init_file:
                    self.encoder.set_hybrid_accu_init_file(self.accu_init_file)
            else:
                raise RuntimeError("Unsupported entropy encoder for decompression")

            self.mapped_quantizer_index = self.encoder.run_decoder(self.compressed_bitstream[-(8 * self.compressed_body_size) :])

            print(f"{time.time() - start_time:.3f} seconds. Done with decoder")
        else:
            print("Using predictor only")

        self.predictor = pred.Predictor(self.header, self.image_constants, self.delayed_weight_updates, self.save_intermediates)

        try:
            decompressed = self.predictor.decompress(self.mapped_quantizer_index)
        except Exception as e:
            self.predictor.save_data(self.output_folder)
            print("Predictor threw exception (", e, "), saving and exiting")
            raise RuntimeError

        print(f"{time.time() - start_time:.3f} seconds. Done with predictor")

        ########################################################################################################################
        # Store result
        ########################################################################################################################

        csv_image_shape = (self.header.y_size * self.header.x_size, self.header.z_size)

        if self.save_intermediates:
            np.savetxt(
                f"{self.output_folder}/predictor-17-decompressed_image_sample.csv",
                decompressed.reshape(csv_image_shape),
                delimiter=",",
                fmt="%d",
            )

        with open(f"{self.output_folder}/z-output-bitstream-dec.bin", "wb") as f:
            if self.header.sample_encoding_order == hd.SampleEncodingOrder.BSQ:
                decompressed = decompressed.transpose(2, 0, 1)  # (z,y,x)
                f.write(decompressed.astype(self.get_sample_format(output_format)).tobytes())
            else:  # BI ordering, including BIP and BIL
                M = self.header.sub_frame_interleaving_depth
                y_size, x_size, z_size = decompressed.shape
                dtype = self.get_sample_format(output_format)

                for y in range(y_size):
                    for i in range((z_size + M - 1) // M):
                        z_start = i * M
                        z_end = min((i + 1) * M, z_size)

                        for x in range(x_size):
                            chunk = decompressed[y, x, z_start:z_end].astype(dtype)
                            f.write(chunk.tobytes())

        print(f"{time.time() - start_time:.3f} seconds. Done with saving")
