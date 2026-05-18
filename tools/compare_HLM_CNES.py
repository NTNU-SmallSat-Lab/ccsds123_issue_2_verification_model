import os
import numpy as np
from pathlib import Path

data_folder = "dw_output"
absolute_error_limits = [0, 1, 3, 7, 15]
num_images = 10


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


def main():
    HLM_avg_data = {}
    compared_data = {}

    for i, pae in enumerate(absolute_error_limits):
        HLM_avg_data[pae] = {}
        HLM_avg_data[pae]["cr"] = 0
        HLM_avg_data[pae]["loading"] = 0
        HLM_avg_data[pae]["predictor"] = 0
        HLM_avg_data[pae]["encoder"] = 0
        HLM_avg_data[pae]["save_data"] = 0

        compared_data[pae] = {}
        compared_data[pae]["CNES"] = 0
        compared_data[pae]["HLM"] = 0

    directory = os.fsencode(data_folder)
    for image_dir in os.listdir(directory):
        image_dir_path = os.path.join(directory, image_dir)
        image_dir_path = image_dir_path.decode()

        HLM_data = np.genfromtxt(image_dir_path + "/results_HLM.txt", delimiter=",")[1:]
        CNES_data = np.genfromtxt(image_dir_path + "/results_CNES.txt", delimiter=",")[1:]

        for i, pae in enumerate(absolute_error_limits):
            HLM_avg_data[pae]["cr"] += HLM_data[i][1]
            HLM_avg_data[pae]["loading"] += HLM_data[i][2]
            HLM_avg_data[pae]["predictor"] += HLM_data[i][3]
            HLM_avg_data[pae]["encoder"] += HLM_data[i][4]
            HLM_avg_data[pae]["save_data"] += HLM_data[i][5]

            compared_data[pae]["CNES"] += CNES_data[i][1]
            compared_data[pae]["HLM"] += HLM_data[i][1]

    for i, pae in enumerate(absolute_error_limits):
        HLM_avg_data[pae]["cr"] /= num_images
        HLM_avg_data[pae]["loading"] /= num_images
        HLM_avg_data[pae]["predictor"] /= num_images
        HLM_avg_data[pae]["encoder"] /= num_images
        HLM_avg_data[pae]["save_data"] /= num_images

        compared_data[pae]["CNES"] /= num_images
        compared_data[pae]["HLM"] /= num_images
        compared_data[pae]["reduction"] = 100 - 100 * compared_data[pae]["CNES"] / compared_data[pae]["HLM"]

    round_leaf_nodes(HLM_avg_data, 3)
    round_leaf_nodes(compared_data, 3)

    print(HLM_avg_data[0])
    print(HLM_avg_data[1])
    print(HLM_avg_data[3])
    print(HLM_avg_data[7])
    print(HLM_avg_data[15])

    print(compared_data[0])
    print(compared_data[1])
    print(compared_data[3])
    print(compared_data[7])
    print(compared_data[15])


if __name__ == "__main__":
    main()
