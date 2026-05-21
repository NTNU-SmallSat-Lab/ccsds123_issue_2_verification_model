import os
import numpy as np

output_folder = "test"
image_folder = "hsi_images"
absolute_error_limits = [0, 1, 3, 7, 15]

hypso_dimensions = {
    "bands": 80,  # z
    "samples": 150,  # x
    "lines": 70,  # y
}
dtype = "<u2"


def round_leaf_nodes(data, decimals=2):
    # Iterate through keys and values
    for key, value in data.items():
        # If the value is a dictionary, recurse deeper
        if isinstance(value, dict):
            round_leaf_nodes(value, decimals)
        # If it's a leaf node (and a number), round it
        elif isinstance(value, (int, float)):
            data[key] = round(value, decimals)
    return data


# reads file of specified format and returns an array formatted as BIP
def format(raw_image_path, dimensions, data_format, ordering="BIP"):
    cube_array = np.fromfile(raw_image_path, dtype=data_format)

    if ordering == "BSQ":
        cube_array = cube_array.reshape(dimensions["bands"], dimensions["lines"], dimensions["samples"])
        cube_array = cube_array.transpose(1, 2, 0)  # transpose to BIP
    elif ordering == "BIL":
        cube_array = cube_array.reshape(dimensions["lines"], dimensions["bands"], dimensions["samples"])
        cube_array = cube_array.transpose(0, 2, 1)  # transpose to BIP
    elif ordering == "BIP":
        cube_array = cube_array.reshape(dimensions["lines"], dimensions["samples"], dimensions["bands"])
    else:
        print(f"Image file ordering {ordering} is unsupported. Suppurted values are 'BSQ', 'BIP' and 'BIL'.")
        raise RuntimeError

    return cube_array


def band_psnr(original, noisy_image, dimensions):
    # Mean Squared Error
    num_pixels = dimensions["lines"] * dimensions["samples"]
    mse_per_band = []
    psnr_per_band = []
    for z in range(dimensions["bands"]):
        mse = np.sum((original[:, :, z] - noisy_image[:, :, z]) ** 2)
        mse /= num_pixels

        psnr = -10 * np.log(mse)
        mse_per_band.append(mse)
        psnr_per_band.append(psnr)

    return psnr_per_band


def main():
    directory = os.fsencode(image_folder)

    PSNR_data = {}
    for pae in absolute_error_limits:
        PSNR_data[pae] = 0

    num_images = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        output_folder_path = f"{output_folder}/{filename.split('.')[0]}"

        print(f"Processing {output_folder_path}")

        original_image = format(f"{image_folder}/{filename}", hypso_dimensions, dtype, "BIP")
        original_image_norm = original_image / np.iinfo(np.dtype(dtype)).max

        for pae in absolute_error_limits:
            try:
                decompressed_image = format(f"{output_folder_path}/{pae}-dec.bin", hypso_dimensions, dtype, "BIP")
            except Exception as e:
                print(f"No image for PAE={pae}")
                continue

            decompressed_image_norm = decompressed_image / np.iinfo(np.dtype(dtype)).max

            if pae == 0:
                assert np.array_equal(original_image, decompressed_image)
                print("Reconstructed image is identical for PAE=0")
                continue

            PSNR = np.average(band_psnr(original_image_norm, decompressed_image_norm, hypso_dimensions))
            PSNR_data[pae] += PSNR

        num_images += 1

    for pae in absolute_error_limits:
        PSNR_data[pae] /= num_images

    round_leaf_nodes(PSNR_data, 3)

    for pae in absolute_error_limits:
        print(f"{pae}: {PSNR_data[pae]}dB")


if __name__ == "__main__":
    main()
