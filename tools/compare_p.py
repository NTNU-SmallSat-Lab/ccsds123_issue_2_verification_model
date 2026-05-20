import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_folder = "p_output"
prediction_bands_HLM = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
prediction_bands_CNES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
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
    compared_data = {}

    for i, P in enumerate(prediction_bands_CNES):
        compared_data[P] = {}
        compared_data[P]["CNES"] = 0
        if P in prediction_bands_HLM:
            compared_data[P]["HLM"] = 0

    directory = os.fsencode(data_folder)
    for image_dir in os.listdir(directory):
        image_dir_path = os.path.join(directory, image_dir)
        image_dir_path = image_dir_path.decode()

        HLM_data = np.genfromtxt(image_dir_path + "/results_HLM.txt", delimiter=",")[1:]
        CNES_data = np.genfromtxt(image_dir_path + "/results_CNES.txt", delimiter=",")[1:]

        for i, P in enumerate(prediction_bands_CNES):
            compared_data[P]["CNES"] += CNES_data[i][1]
            if P in prediction_bands_HLM:
                compared_data[P]["HLM"] += HLM_data[prediction_bands_HLM.index(P)][1]

    for i, P in enumerate(prediction_bands_CNES):
        compared_data[P]["CNES"] /= num_images
        if P in prediction_bands_HLM:
            compared_data[P]["HLM"] /= num_images

    round_leaf_nodes(compared_data, 3)

    cr_CNES = [compared_data[P]["CNES"] for P in prediction_bands_CNES]
    cr_HLM = [compared_data[P]["HLM"] for P in prediction_bands_HLM]

    print(cr_CNES)
    print(cr_HLM)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Sets the math font to Computer Modern
        }
    )

    plt.rcParams["font.size"] = 14

    # To apply it to regular text as well, use the internal 'cmr10' font name
    plt.rcParams["font.serif"] = "cmr10"
    plt.rcParams["axes.unicode_minus"] = False  # Fixes potential minus sign issues

    plt.plot(prediction_bands_CNES[:-3], cr_CNES[:-3], color="royalblue", marker="x", linewidth="1", label="Ordinary")
    plt.plot(prediction_bands_HLM, cr_HLM, color="tomato", marker="*", linewidth="1", linestyle="-", label="Delayed")
    plt.ylabel("CR")
    plt.xlabel("Number of Prediction Bands")
    plt.title("Coastline HYPSO-2")
    plt.ylim((2.4, 2.6))
    plt.tick_params(axis="both", direction="in")
    plt.yticks((2.40, 2.45, 2.50, 2.55, 2.60))
    plt.xticks(prediction_bands_CNES[:-3])
    plt.legend()
    plt.grid(linestyle=(0, (3, 6)))
    plt.tight_layout()
    plt.savefig("prediction_bands.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
