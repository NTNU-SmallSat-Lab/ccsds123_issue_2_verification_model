from ccsds123_i2_hlm import header as hd
import os
import subprocess
from pathlib import Path

image_folder = "/Users/aasmundnorsett/Documents/NTNU/SmallSat/resources/hsi-images/hypso2-raw"
ccsds_output_folder = str(Path(__file__).resolve().parent) + "/ccsds_output"
# absolute_error_limits = [0, 1, 3, 7, 15]
prediction_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def main():
    directory = os.fsencode(image_folder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filepath = os.path.join(directory, file)
        output_folder = f"p_output/{filename.split('.')[0]}"
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        with open(f"{output_folder}/results_CNES.txt", "a") as file:
            file.truncate(0)
            file.write("P,cr\n")

        print(filename)

        # for pae in absolute_error_limits:
        for P in prediction_bands:
            # print(f"Compressing with PAE {pae}")
            # ccsds.header.set_absolute_error_limit_value(pae)

            # print(f"Compressing with PAE {pae}")
            print(f"Compressing with {P} prediction bands")

            header = hd.Header(filename)
            header.sub_frame_interleaving_depth = header.z_size
            # header.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY
            # header.set_absolute_error_limit_value(pae)
            header.prediction_bands_num = P
            header.save_data(ccsds_output_folder)

            os.environ["IMAGE_DIR"] = image_folder
            os.environ["CCSDS_OUTPUT"] = ccsds_output_folder

            status = subprocess.call(
                [
                    str(Path(__file__).resolve().parent) + "/script/cnes_encode.sh",
                    "header.bin",
                    "u16le",
                    filename,
                    "z-output-bitstream-enc.bin",
                ]
            )

            if status != 0:
                exit(1)

            cr = os.path.getsize(filepath) / os.path.getsize(ccsds_output_folder + "/z-output-bitstream-enc.bin")

            with open(f"{output_folder}/results_CNES.txt", "a") as file:
                file.write(f"{P},{cr:.3f}\n")

            os.rename(ccsds_output_folder + "/z-output-bitstream-enc.bin", f"{output_folder}/{P}.bin")


if __name__ == "__main__":
    main()
