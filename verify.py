from src import ccsds123
import os
import keyboard

test_vector_folder = "../Test1-20190201"


def main():
    test_vector_files = os.listdir(test_vector_folder)

    input_raw_files = [file for file in test_vector_files if file.endswith(".raw")]
    input_header_files = [file for file in test_vector_files if file.endswith("hdr.bin")]
    input_optional_tables = [file for file in test_vector_files if file.endswith("optional_tables.bin")]
    input_error_limits = [file for file in test_vector_files if file.endswith("error_limits.bin")]
    input_hybrid_tables = [file for file in test_vector_files if file.endswith("hybrid_initial_accumulators.bin")]
    golden_compressed_files = [file for file in test_vector_files if file.endswith(".flex")]


    for num in range(6, len(input_raw_files)):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Test {num}")
        print(f"Input raw file: {input_raw_files[num]}")
        print(f"Input header file: {input_header_files[num]}")
        print(f"Input optional tables file: {input_optional_tables[num]}")
        print(f"Input error limits file: {input_error_limits[num]}")
        print(f"Input hybrid tables file: {input_hybrid_tables[num]}")
        print(f"Golden compressed file: {golden_compressed_files[num]}")

        print(f"For more debug data, run: ")
        print(f"make compare_vector image={test_vector_folder}/{input_raw_files[num]} header={test_vector_folder}/{input_header_files[num]} image_format=s32be correct={test_vector_folder}/{golden_compressed_files[num]} optional_tables={test_vector_folder}/{input_optional_tables[num]} error_limits={test_vector_folder}/{input_error_limits[num]} accu={test_vector_folder}/{input_hybrid_tables[num]} ")
        print(f"header_tool -t {test_vector_folder}/{input_optional_tables[num]} -d {test_vector_folder}/{input_header_files[num]}")

        dut_compressor = ccsds123.CCSDS123(f"{test_vector_folder}/{input_raw_files[num]}")
        dut_compressor.set_header_file(f"{test_vector_folder}/{input_header_files[num]}")
        dut_compressor.set_optional_tables_file(f"{test_vector_folder}/{input_optional_tables[num]}")
        dut_compressor.set_error_limits_file(f"{test_vector_folder}/{input_error_limits[num]}")
        dut_compressor.set_hybrid_accu_init_file(f"{test_vector_folder}/{input_hybrid_tables[num]}")

        dut_compressor.compress_image()

        with open("output/z-output-bitstream.bin", 'rb') as file1, \
            open(f"{test_vector_folder}/{golden_compressed_files[num]}", 'rb') as file2:
            content1 = file1.read()
            content2 = file2.read()

        if content1 == content2:
            print(f"Files in test {num} are identical")
        else:
            print(f"Files in test {num} are different")
            

            exit()


if __name__ == "__main__":
    main()
