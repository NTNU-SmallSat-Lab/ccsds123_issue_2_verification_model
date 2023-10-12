import numpy as np
import re
from bitarray import bitarray

image_name = 'Landsat_mountain-u16be-6x1024x1024.raw'
new_x_size = 100
new_y_size = 50
new_z_size = 6


raw_image_folder = "raw_images"


def main():
    x_size = int(re.findall('x(.*).raw', image_name)[0].split("x")[-1]) 
    y_size = int(re.findall('x(.+)x', image_name)[0])
    z_size = int(re.findall('-(.+)x', re.findall('-(.+)x', image_name)[0])[0])

    bits = 16

    file = open(raw_image_folder + "/" + image_name, 'rb').read()
    first = file[0] # Set aside first byte
    last = (file[-2] << 8) + file[-1] # set aside last two bytes
    with open(raw_image_folder + "/" + image_name, 'rb') as file:
        file.seek(1) # Skip the first byte
        image_sample = np.fromfile(file, dtype=np.uint16)
    image_sample=np.pad(image_sample, (0,1), 'constant', constant_values=last) # Pad the last two bytes
    image_sample[0] += first << 8 # Reintroduce the first byte
    image_sample = image_sample.reshape((z_size, y_size, x_size))

    new_image = image_sample[:new_z_size, :new_y_size, :new_x_size]

    new_image = new_image.reshape(new_z_size * new_y_size * new_x_size)
    # print(new_image.shape)

    new_image_name = raw_image_folder + "/" + image_name.split(f"-{z_size}x")[0] + f"-{new_z_size}x{new_y_size}x{new_x_size}.raw"

    # np.save(new_image_name, new_image)
    # print(new_image[:10])
    # print(new_image.tobytes()[:10])
    # with open(new_image_name, 'wb') as file:
    #     # file.write(bytes([0]))
    #     file.write(new_image.tobytes())

    bitstream = bitarray()
    for sample in new_image:
        bitstring = bin(sample)[2:]
        bitstream += '0' * (bits - len(bitstring)) + bitstring

    with open(new_image_name, "wb") as file:
            bitstream.tofile(file)

    print(f"Saved new image as {new_image_name}")

if __name__ == "__main__":
    main()

