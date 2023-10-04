from src import header as hd
from src import ccsds123
import numpy as np
import time

image_name = 'aviris_maine_f030828t01p00r05_sc10.uncal-u16be-224x512x680.raw'
image_name = 'Landsat_mountain-u16be-6x1024x1024.raw'



def main():
    start_time = time.time()

    image = ccsds123.CCSDS123(image_name)
    image.load_raw_image()

    print(image.image_raw.shape)
    print(image.image_raw)

    elapsed_time = time.time() - start_time
    print(f"Done! Script ran for {elapsed_time:.3f} seconds")


if __name__ == "__main__":
    main()