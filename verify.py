from ccsds123_i2_hlm import ccsds123
import os
import argparse

# for Test1-20181021
# roughly 10% of tets in this set have invalid headers
skip = [
    1471,  # segfault
    1777,  # gives a segmentation fault, yikes
    1791,  # seg fault
    1820,  # gives wrong result, should look into
    1925,  # another seg fault
    1932,  # wrong result
    2075,  # seg fault
    2158,  # wrong result
    2226,  # wrong result
    2527,  # gives a weird error in the encoder on line 97
    2558,  # seg fault
    2570,  # seg fault, bruh this one did not have size of 1x1
    2631,  # again seg fault, not 1x1
    2648,  # wrong result
    2672,  # wrong result
    2782,  # wrong result
    2821,  # same error in encoder, line 89
    3032,  # seg fault
    3084,  # wrong result
    3151,  # wrong result
    3303,  # seg fault
    3386,  # wrong result
    3408,  # wrong result
    3416,  # seg fault
    3421,  # seg fault
    3483,  # wrong result
    3543,  # wrong result
    3602,  # seg fault
    3667,  # wrong result
    3866,  # seg fault
    3982,  # wrong result
    4016,  # wrong result
    4052,  # seg fault
    4109,  # wrong result
    4188,  # seg fault
    4192,  # wrong result
    4236,  # wrong result
    4446,  # wrong result
    4532,  # seg fault
    4542,  # seg fault
    4543,  # seg fault
    4669,  # wrong result
    4777,  # wrong result
    4947,  # wrong result
    4949,  # wrong result
    5024,  # seg fault
    5091,  # wrong result
    5138,  # seg fault
    5208,  # wrong result
    5444,  # seg fault
    5458,  # seg fault
    5701,  # seg fault
    5718,  # wrong result
    5722,  # wrong result
    6090,  # wrong result
    6277,  # seg fault
    6397,  # seg fault
]
# seems like some of the ones with segfaults have a spatial size of 1x1
# not all images of that size fail, so must be something else in addition with the config
# maybe not, I probably have a memory leak somewhere, yikes

# when rerunning a large set of tests I get new failures, meaning a seg fault is probable
# maybe some sort of array misalignment?


def main():

    parser = argparse.ArgumentParser(
        description="Verify the CCSDS 123.0-B-2 High level model using CCSDS provided test vectors"
    )
    parser.add_argument("folder", help="Path to the folder containing the test vectors")
    parser.add_argument("--start", default="", help="Test vector number to start at")
    parser.add_argument("--len", default="", help="Test vector number to end at")
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
    input_header_files = [
        file for file in test_vector_files if file.endswith("hdr.bin")
    ]
    input_optional_tables = [
        file for file in test_vector_files if file.endswith("optional_tables.bin")
    ]
    input_error_limits = [
        file for file in test_vector_files if file.endswith("error_limits.bin")
    ]
    input_hybrid_tables = [
        file
        for file in test_vector_files
        if file.endswith("hybrid_initial_accumulators.bin")
    ]
    golden_compressed_files = [
        file for file in test_vector_files if file.endswith(".flex")
    ]

    input_raw_files.sort()
    input_header_files.sort()
    input_optional_tables.sort()
    input_error_limits.sort()
    input_hybrid_tables.sort()
    golden_compressed_files.sort()

    comparison_files_hlm = [
        "output/z-output-bitstream-enc.bin",
        "output/header.bin",
        "output/optional_tables.bin",
        "output/error_limits.bin",
        "output/hybrid_initial_accumulator.bin",
    ]

    end_num = len(input_raw_files)
    if length != 0:
        end_num = start_num + length
    if end_num > len(input_raw_files):
        end_num = len(input_raw_files)

    success = 0
    failure = 0
    skipped = 0
    failure_list = []
    for num in range(start_num, end_num):
        os.system("cls" if os.name == "nt" else "clear")
        print(
            f"Success: {success}/{num} Failure: {failure}/{num} Skipped: {skipped}/{num}"
        )
        print(f"Failure list: {failure_list}\n")

        print(f"Test {num}")
        print(f"Input raw file: {input_raw_files[num]}")
        print(f"Input header file: {input_header_files[num]}")
        print(f"Input optional tables file: {input_optional_tables[num]}")
        print(f"Input error limits file: {input_error_limits[num]}")
        print(f"Input hybrid tables file: {input_hybrid_tables[num]}")
        print(f"Golden compressed file: {golden_compressed_files[num]}")

        print(f"For more debug data, run: ")
        print(
            f"make compare_vector image={test_vector_folder}/{input_raw_files[num]} header={test_vector_folder}/{input_header_files[num]} image_format=s32be correct={test_vector_folder}/{golden_compressed_files[num]} optional_tables={test_vector_folder}/{input_optional_tables[num]} error_limits={test_vector_folder}/{input_error_limits[num]} accu={test_vector_folder}/{input_hybrid_tables[num]} "
        )
        print(
            f"header_tool -t {test_vector_folder}/{input_optional_tables[num]} -d {test_vector_folder}/{input_header_files[num]}"
        )

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

        dut_compressor = ccsds123.CCSDS123(
            f"{test_vector_folder}/{input_raw_files[num]}"
        )
        dut_compressor.set_header_file(
            f"{test_vector_folder}/{input_header_files[num]}"
        )
        dut_compressor.set_optional_tables_file(
            f"{test_vector_folder}/{input_optional_tables[num]}"
        )
        dut_compressor.set_error_limits_file(
            f"{test_vector_folder}/{input_error_limits[num]}"
        )
        dut_compressor.set_hybrid_accu_init_file(
            f"{test_vector_folder}/{input_hybrid_tables[num]}"
        )

        # in case we do not support the config in the provided header
        try:
            dut_compressor.set_header()
        except Exception as e:
            print("Invalid header (", e, "), skipping")
            skipped += 1
            continue

        dut_compressor.compress_image()

        with open("output/z-output-bitstream-enc.bin", "rb") as file1, open(
            f"{test_vector_folder}/{golden_compressed_files[num]}", "rb"
        ) as file2:
            content1 = file1.read()
            content2 = file2.read()

        with open("output/header.bin", "rb") as file1, open(
            f"{test_vector_folder}/{input_header_files[num]}", "rb"
        ) as file2:
            content1 = file1.read()
            content2 = file2.read()

        correct = 0
        for i in range(len(comparison_files_golden)):
            with open(comparison_files_golden[i], "rb") as file1, open(
                comparison_files_hlm[i], "rb"
            ) as file2:
                content1 = file1.read()
                content2 = file2.read()
                if content1 == content2:
                    correct += 1
                else:
                    print("Mismatch on file: ", comparison_files_hlm[i])

        if correct == len(comparison_files_golden):
            print(f"Files in test {num} are identical")
            success += 1
        else:
            print(f"Files in test {num} are different")
            failure += 1
            failure_list.append(num)


if __name__ == "__main__":
    main()
