from ccsds123_i2_hlm import ccsds123
import time
import os
import psutil
import argparse
from pathlib import Path

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024) # MegaBytes

def get_file_size(file_path):
    return os.path.getsize(file_path)

def main():
    parser = argparse.ArgumentParser(description="Compress an image using CCSDS 123.0-B-2 and produce intermediate files for debugging")
    parser.add_argument("image_file", help="Path to the raw uncompressed image file. The filename must be on the format <name>-<datatype>-<z_size>x<y_size>x<x_size>.raw like CCSDS TestData images. For example Landsat_mountain-u16be-6x50x100.raw.")
    parser.add_argument("--header", default="", help="Path to the CCSDS 123.0-B-2 header binary file used to set compression settings. The header is formatted as it is in a CCSDS 123.0-B-2 compressed image.")
    parser.add_argument("--accu" , default="", help="Path to the hybrid encoder accumulator initial values binary file. Stored as unsigned integers in increasing band order, using D+gamma_0 bits, in a file that is zero padded to the nearest byte at the end.")
    parser.add_argument("--optional", default="", help="Path to the optional tables binary file. These are tables that could also be stored in the header. Values are stored as they would be in the header. Tables are stored in the order they would be in the header.")
    parser.add_argument("--error_limits", default="", help="Path to the error limits binary file for when using periodic error limit updating. Values are stored as 16-bit unsigned integers in the same order they would be in the image.")
    args = parser.parse_args()

    start_time = time.time()

    image = ccsds123.CCSDS123(args.image_file)
    if len(args.header) > 0:
        image.set_header_file(args.header)
    if len(args.accu) > 0:
        image.set_hybrid_accu_init_file(args.accu)
    if len(args.optional) > 0:
        image.set_optional_tables_file(args.optional)
    if len(args.error_limits) > 0:
        image.set_error_limits_file(args.error_limits)

    image.compress_image()

    elapsed_time = time.time() - start_time
    print(f"Done! Script ran for {elapsed_time:.3f} seconds")
    print(f"Memory usage: {get_memory_usage():.2f} MB")
    print(f"Compression ratio: {get_file_size(args.image_file) / get_file_size(str(Path(__file__).resolve().parent) + '/output/z-output-bitstream.bin'):.2f}")
    

if __name__ == "__main__":
    main()
