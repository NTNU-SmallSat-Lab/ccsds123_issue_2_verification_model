from ccsds123_i2_hlm import ccsds123
from ccsds123_i2_hlm import header as hd
import os
from pathlib import Path

image_folder = "hsi_images"
hlm_output_folder = str(Path(__file__).resolve().parent) + "/hlm_output"
absolute_error_limits = [0, 1, 3, 7, 15]
# prediction_bands = [1, 2, 3, 7, 12]


def main():
    ccsds = ccsds123.CCSDS123(image_ordering="BIP", delayed_weight_updates=True)
    ccsds.set_output_dir(hlm_output_folder)
    ccsds.set_header()

    ccsds.header.z_size = 80
    ccsds.header.y_size = 70
    ccsds.header.x_size = 150
    # ccsds.header.sub_frame_interleaving_depth = ccsds.header.z_size
    ccsds.header.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY

    directory = os.fsencode(image_folder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        # filepath = os.path.join(directory, file)
        output_folder = f"test/{filename.split('.')[0]}"
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        with open(f"{output_folder}/results_HLM-dec.txt", "a") as file:
            file.truncate(0)
            # file.write("pae,pr,loading,predictor,encoder,save_data\n")
            file.write("pae,loading,predictor,encoder,save_data\n")

        for pae in absolute_error_limits:
            # for P in prediction_bands:
            print(f"Compressing with PAE {pae}")
            ccsds.header.set_absolute_error_limit_value(pae)

            # print(f"Compressing with {P} prediction bands")
            # ccsds.header.prediction_bands_num = P

            # timing = ccsds.compress_image(filepath.decode(), file_format="u16le")
            timing = ccsds.decompress_image(f"{output_folder}/{pae}.bin", output_format="u16le")

            # cr = os.path.getsize(filepath) / os.path.getsize(hlm_output_folder + "/z-output-bitstream-enc.bin")

            with open(f"{output_folder}/results_HLM.txt", "a") as file:
                # file.write(f"{pae},{cr:.3f},{timing['loading']:.3f},{timing['predictor']:.3f},{timing['encoder']:.3f},{timing['save_data']:.3f}\n")
                file.write(f"{pae},{timing['loading']:.3f},{timing['predictor']:.3f},{timing['encoder']:.3f},{timing['save_data']:.3f}\n")

            # os.rename(hlm_output_folder + "/z-output-bitstream-enc.bin", f"{output_folder}/{pae}.bin")
            os.rename(hlm_output_folder + "/z-output-bitstream-dec.bin", f"{output_folder}/{pae}-dec.bin")


if __name__ == "__main__":
    main()
