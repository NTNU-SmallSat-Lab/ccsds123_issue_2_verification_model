from src import header as hd
import numpy as np
import time

image_name = 'aviris_maine_f030828t01p00r05_sc10.uncal-u16be-224x512x680.raw'
image_name = 'Landsat_mountain-u16be-6x1024x1024.raw'

raw_image_folder = "raw_images"

def main():
    start_time = time.time()


    header = hd.Header(image_name)

    # Load a raw image into a N_x * N_y by N_z array
    # Doing it the obvious way skipped the first byte, hence some hoops have been jumped through to fix it
    file = open(raw_image_folder + "/" + image_name, 'rb').read()
    first = file[0]
    last = (file[-2] << 8) + file[-1]
    with open(raw_image_folder + "/" + image_name, 'rb') as f:
        f.seek(1)
        image_raw = np.fromfile(f, dtype=np.uint16)
    image_raw=np.pad(image_raw, (0,1), 'constant', constant_values=last)
    image_raw[0] += first << 8
    image_raw.shape = (header.z_size, image_raw.size//header.z_size)
    print(image_raw.shape)
    print(image_raw)
    






    elapsed_time = time.time() - start_time
    print(f"Done! Script ran for {elapsed_time:.2f} seconds")



if __name__ == "__main__":
    main()