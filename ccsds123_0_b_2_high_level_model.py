from src import header as hd
from src import ccsds123
import numpy as np
import time
import os
import psutil
import argparse


image_file = 'raw_images/aviris_maine_f030828t01p00r05_sc10.uncal-u16be-224x512x680.raw'
image_file = 'raw_images/Landsat_mountain-u16be-6x100x100.raw'
image_file = 'raw_images/Landsat_mountain-u16be-6x100x200.raw'
image_file = 'raw_images/Landsat_mountain-u16be-6x1024x1024.raw'
image_file = 'raw_images/Landsat_mountain-u16be-6x50x100.raw'

# Function to get memory usage in MB
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)

def main():
    parser = argparse.ArgumentParser(description="Compress an image using CCSDS 123.0-B-2 and produce intermediate files for debugging")
    parser.add_argument("image_file", help="Path to the image file.")
    parser.add_argument("--header", default="", help="Path to the header file.")
    parser.add_argument("--accu" , default="", help="Path to the hybrid encoder accumulator initial values file. Stored as unsigned integers in increasing band order, using D+gamma_0 bits, in a file that is zero padded to the nearest byte at the end.")
    parser.add_argument("--optional", default="", help="Path to the optional tables file. These are tables that could also be stored in the header. Values are stored as they would be in the header. Tables are stored in the order they would be in the header.")
    parser.add_argument("--error_limits", default="", help="Path to the error limits file for when using periodic error limit updating. Values are stored as 16-bit unsigned integers in the same order they would be in the header.")
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


if __name__ == "__main__":
    main()