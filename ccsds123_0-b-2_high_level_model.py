from src import header as hd
from src import ccsds123
import numpy as np
import time
import os
import psutil

image_name = 'aviris_maine_f030828t01p00r05_sc10.uncal-u16be-224x512x680.raw'
image_name = 'Landsat_mountain-u16be-6x1024x1024.raw'

# Function to get memory usage in MB
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)

def main():
    start_time = time.time()

    image = ccsds123.CCSDS123(image_name)
    image.load_raw_image()

    # print(image.image_sample.shape)
    # print(image.image_sample)

    image.predictor()

    elapsed_time = time.time() - start_time
    print(f"Done! Script ran for {elapsed_time:.3f} seconds")
    print(f"Memory usage: {get_memory_usage():.2f} MB")


if __name__ == "__main__":
    main()