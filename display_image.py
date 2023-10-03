import rawpy
import imageio
from matplotlib import pyplot as plt
from PIL import Image
import re

imageName = 'Landsat_mountain-u16be-6x1024x1024.raw'
imageName = 'aviris_hawaii_f011020t01p03r05_sc01.uncal-u16be-224x512x614.raw'
imageName = 'aviris_maine_f030828t01p00r05_sc10.uncal-u16be-224x512x680.raw'


# I am a regex master
imageXcount = int(re.findall('x(.*).raw', imageName)[0].split("x")[-1]) 
imageYcount = int(re.findall('x(.+)x', imageName)[0])
imageZcount = int(re.findall('-(.+)x', re.findall('-(.+)x', imageName)[0])[0])
imageType = re.findall('-(.*)-', imageName)[0]

print(imageXcount)
print(imageYcount)
print(imageZcount)
print(imageType)

rawFolder = "raw_images"

path = rawFolder + "/" + imageName
f = open(path, 'rb').read()
for i in range(40):
    print(f[i], end=" ")

# print("")
# offset = 2*1024
# for i in range(offset, offset+40):
#     print(f[i], end=" ")



index = 0 + 2 * imageXcount * imageYcount * 128
# factor = 1

img = Image.new('RGB', (imageXcount, imageYcount), "black")  # Create a new black image
pixels = img.load()  # Create the pixel map
for y in range(img.size[0]):    # For every pixel:
    for x in range(img.size[1]):
        if imageType == "u16be":
            pixelVal = (f[index] << 8) + f[index+1]
        else:
            exit("Unknown image type")
        pixels[x, y] = (pixelVal,pixelVal,pixelVal)
        index += 2

img.show()
