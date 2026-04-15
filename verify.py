from numpy import save
from ccsds123_i2_hlm import ccsds123, predictor_old
from ccsds123_i2_hlm import header as hd
import os
import argparse

skip = []

delayed_weight_updates = False
save_intermediates = False
use_old_predictor = False
predictor_only = False

# Fails
# CPP compressor:            [32, 164, 218, 804, 1510, 1871]                                           --> these are shared between cpp predictor compress/decompress
# Decompress only predictor: [32, 53, 164, 218, 313, 464, 466, 585, 804, 1115, 1510, 1842, 1871, 1927] --> the extra fails are only in decompression part of predictor
# Full decompressor:         [464, 585]                                                                --> Only hybrid encoder


def main():
    parser = argparse.ArgumentParser(description="Verify the CCSDS 123.0-B-2 High level model using CCSDS provided test vectors")
    parser.add_argument("folder", help="Path to the folder containing the test vectors")
    parser.add_argument("--start", default="", help="Test vector number to start at")
    parser.add_argument("--len", default="", help="Test vector number to end at")
    parser.add_argument("-d", "--decompress", action="store_true", default=False, help="Add to Verify decompression, only supports hybrid encoder")
    args = parser.parse_args()

    start_num = 0
    length = 0
    test_vector_folder = args.folder
    if len(args.start) > 0:
        start_num = int(args.start)
    if len(args.len) > 0:
        length = int(args.len)

    test_vector_files = os.listdir(test_vector_folder)

    input_raw_files = [file for file in test_vector_files if file.endswith(".raw")]
    input_header_files = [file for file in test_vector_files if file.endswith("hdr.bin")]
    input_optional_tables = [file for file in test_vector_files if file.endswith("optional_tables.bin")]
    input_error_limits = [file for file in test_vector_files if file.endswith("error_limits.bin")]
    input_hybrid_tables = [file for file in test_vector_files if file.endswith("hybrid_initial_accumulators.bin")]
    golden_compressed_files = [file for file in test_vector_files if file.endswith(".flex")]
    golden_decompressed_files = [file for file in test_vector_files if file.endswith("dec.bin")]

    input_raw_files.sort()
    input_header_files.sort()
    input_optional_tables.sort()
    input_error_limits.sort()
    input_hybrid_tables.sort()
    golden_compressed_files.sort()
    golden_decompressed_files.sort()

    comparison_files_hlm = [
        "output/z-output-bitstream-enc.bin",
        "output/header.bin",
        "output/optional_tables.bin",
        "output/error_limits.bin",
        "output/hybrid_initial_accumulator.bin",
    ]

    if args.decompress:
        comparison_files_hlm.append("output/z-output-bitstream-dec.bin")

    end_num = len(input_raw_files)
    if length != 0:
        end_num = start_num + length
    if end_num > len(input_raw_files):
        end_num = len(input_raw_files)

    success = 0
    failure = 0
    skipped = 0
    failure_list = []
    skipped_list = []
    success_list = []
    not_skipped_list = []
    for num in range(start_num, end_num):
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Success: {success}/{num} Failure: {failure}/{num} Skipped: {skipped}/{num}")
        print(f"Failure list: {failure_list}\n")
        # print(f"Success list: {success_list}\n")
        # print(f"Skipped list: {skipped_list}\n")
        # print(f"Not Skipped list: {not_skipped_list}\n")

        print(f"Test {num}")
        print(f"Input raw file: {input_raw_files[num]}")
        print(f"Input header file: {input_header_files[num]}")
        print(f"Input optional tables file: {input_optional_tables[num]}")
        print(f"Input error limits file: {input_error_limits[num]}")
        print(f"Input hybrid tables file: {input_hybrid_tables[num]}")
        print(f"Golden compressed file: {golden_compressed_files[num]}")
        print(f"Golden decompressed file: {golden_decompressed_files[num]}")

        print(f"For more debug data, run: ")
        print(f"make compare_vector image={test_vector_folder}/{input_raw_files[num]} header={test_vector_folder}/{input_header_files[num]} image_format=s32be correct={test_vector_folder}/{golden_compressed_files[num]} optional_tables={test_vector_folder}/{input_optional_tables[num]} error_limits={test_vector_folder}/{input_error_limits[num]} accu={test_vector_folder}/{input_hybrid_tables[num]} ")
        print(f"header_tool -t {test_vector_folder}/{input_optional_tables[num]} -d {test_vector_folder}/{input_header_files[num]}")

        if num in skip:
            skipped += 1
            continue

        comparison_files_golden = [
            f"{test_vector_folder}/{golden_compressed_files[num]}",
            f"{test_vector_folder}/{input_header_files[num]}",
            f"{test_vector_folder}/{input_optional_tables[num]}",
            f"{test_vector_folder}/{input_error_limits[num]}",
            f"{test_vector_folder}/{input_hybrid_tables[num]}",
        ]
        if args.decompress:
            comparison_files_golden.append(f"{test_vector_folder}/{golden_decompressed_files[num]}")

        dut_compressor = ccsds123.CCSDS123(delayed_weight_updates=delayed_weight_updates, save_intermediates=save_intermediates, use_old_predictor=use_old_predictor, predictor_only=predictor_only)
        dut_compressor.set_header_file(f"{test_vector_folder}/{input_header_files[num]}")
        dut_compressor.set_optional_tables_file(f"{test_vector_folder}/{input_optional_tables[num]}")
        dut_compressor.set_error_limits_file(f"{test_vector_folder}/{input_error_limits[num]}")
        dut_compressor.set_hybrid_accu_init_file(f"{test_vector_folder}/{input_hybrid_tables[num]}")

        dut_compressor.set_header()  # so that we can check header config values
        if not predictor_only and args.decompress and dut_compressor.header.entropy_coder_type != hd.EntropyCoderType.HYBRID:
            skipped += 1
            skipped_list.append(num)
            print("Skipping")
            continue

        dut_compressor.compress_image(f"{test_vector_folder}/{input_raw_files[num]}")

        if args.decompress:
            dut_compressor.decompress_image(comparison_files_golden[0])

        correct = 0
        for i in range(len(comparison_files_golden)):
            with open(comparison_files_golden[i], "rb") as file1, open(comparison_files_hlm[i], "rb") as file2:
                content1 = file1.read()
                content2 = file2.read()

                if content1 == content2:
                    correct += 1
                else:
                    print("Mismatch on file: ", comparison_files_hlm[i])

        not_skipped_list.append(num)
        if correct == len(comparison_files_golden):
            print(f"Files in test {num} are identical")
            success += 1
            success_list.append(num)
        else:
            print(f"Files in test {num} are different")
            failure += 1
            failure_list.append(num)


if __name__ == "__main__":
    main()
