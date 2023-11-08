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
    parser.add_argument("image_file", help="Path to the image file")
    parser.add_argument("--header", default="", help="Path to the configuration file")
    parser.add_argument("--accu" , default="", help="Path to the hybrid encoder accumulator initial values file")
    args = parser.parse_args()

    start_time = time.time()

    image = ccsds123.CCSDS123(args.image_file)
    if len(args.header) > 0:
        image.set_header_file(args.header)
    if len(args.accu) > 0:
        image.set_hybrid_accu_init_file(args.accu)

    image.compress_image()

    elapsed_time = time.time() - start_time
    print(f"Done! Script ran for {elapsed_time:.3f} seconds")
    print(f"Memory usage: {get_memory_usage():.2f} MB")


if __name__ == "__main__":
    main()