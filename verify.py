from src import ccsds123
import os
import keyboard

test_vector_folder = "Test1-20190201"


def main():
    test_vector_files = os.listdir(test_vector_folder)

    input_raw_files = [file for file in test_vector_files if file.endswith(".raw")]
    input_header_files = [file for file in test_vector_files if file.endswith("hdr.bin")]
    golden_compressed_files = [file for file in test_vector_files if file.endswith(".flex")]

    for num in range(len(input_raw_files)):
        try:
            dut_compressor = ccsds123.CCSDS123(f"{test_vector_folder}/{input_raw_files[num]}")
            dut_compressor.set_header_file(f"{test_vector_folder}/{input_header_files[num]}")
            dut_compressor.compress_image()
        except:
            print(f"Test {num} failed")
            if keyboard.is_pressed('q'):
                exit()

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
