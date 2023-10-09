from PIL import Image
import re
import numpy as np

display_layer = 1 # 1-indexed
# image_name = 'aviris_hawaii_f011020t01p03r05_sc01.uncal-u16be-224x512x614.raw'
image_name = 'aviris_maine_f030828t01p00r05_sc10.uncal-u16be-224x512x680.raw'
image_name = 'Landsat_mountain-u16be-6x100x100.raw'
image_name = 'Landsat_mountain-u16be-6x1024x1024.raw'


# I am a regex master
x_size = int(re.findall('x(.*).raw', image_name)[0].split("x")[-1]) 
y_size = int(re.findall('x(.+)x', image_name)[0])
z_size = int(re.findall('-(.+)x', re.findall('-(.+)x', image_name)[0])[0])
imageType = re.findall('-(.*)-', image_name)[0]

# print(x_size)
# print(y_size)
# print(z_size)
# print(imageType)

raw_image_folder = "raw_images"


file = open(raw_image_folder + "/" + image_name, 'rb').read()
first = file[0] # Set aside first byte
last = (file[-2] << 8) + file[-1] # set aside last two bytes
with open(raw_image_folder + "/" + image_name, 'rb') as file:
    file.seek(1) # Skip the first byte
    image_sample = np.fromfile(file, dtype=np.uint16)
if image_sample.shape[0] != z_size * y_size * x_size:
    image_sample=np.pad(image_sample, (0,1), 'constant', constant_values=last) # Pad the last two bytes
image_sample[0] += first << 8 # Reintroduce the first byte
image_sample = image_sample.reshape((z_size, y_size, x_size))


img = Image.new('RGB', (x_size, y_size), "black")  # Create a new black image
pixels = img.load()  # Create the pixel map

for y in range(img.size[1]):    # For every pixel:
    for x in range(img.size[0]):
        val = image_sample[display_layer-1, y, x]
        pixels[x, y] = (val,val,val)

img.show()
