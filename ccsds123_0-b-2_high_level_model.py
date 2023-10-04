from src import header as hd
import numpy as np
import time

image_name = 'Landsat_mountain-u16be-6x1024x1024.raw'
image_name = 'aviris_maine_f030828t01p00r05_sc10.uncal-u16be-224x512x680.raw'

raw_image_folder = "raw_images"

def main():
    start_time = time.time()


    header = hd.Header(image_name)

    file = open(raw_image_folder + "/" + image_name, 'rb').read()
    first = file[0]
    last = (file[-2] << 8) + file[-1]
    # file.close()
    # for i in range(40):
    #     print(file[i], end=" ")

    # image_raw = np.empty((header.z_size, header.x_size*header.y_size), dtype=np.uint16)
    # index = 0
    # for z in range(header.z_size):
    #     for y in range(header.y_size):
    #         for x in range(header.x_size):
    #             image_raw[z][y*header.x_size + x] = (file[index] << 8) + file[index+1]
    #             index += 2
    # print(image_raw.shape)
    # print(image_raw)

    with open(raw_image_folder + "/" + image_name, 'rb') as f:
    # Read the binary data into a NumPy array
        f.seek(1)
        image_raw = np.fromfile(f, dtype=np.uint16)
        # print(f[0])
    image_raw=np.pad(image_raw, (0,1), 'constant', constant_values=last)
    image_raw[0] += first<<8
    print(image_raw.shape)
    print(image_raw)
    image_raw.shape = (header.z_size, image_raw.size//header.z_size)
    print(image_raw.shape)
    print(image_raw)
    # image_raw_matrix = np.reshape(image_raw, (-1, 2))
    






    elapsed_time = time.time() - start_time
    print(f"Done! Script ran for {elapsed_time:.2f} seconds")



if __name__ == "__main__":
    main()