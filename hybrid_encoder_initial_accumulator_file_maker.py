import argparse
from bitarray import bitarray


def main():
    parser = argparse.ArgumentParser(description="Create a hybrid encoder initial accumulator value file.")
    parser.add_argument("output_file", help="File to output to.")
    parser.add_argument("accumulators", help="Number of accumulators. Same as N_Z in the header file.")
    parser.add_argument("accumulator_size_bits", help="Number of bits each accumulator is stored with. Same as D + gamma_0.")
    parser.add_argument("value", help="Value to assign to all accumulators.")
    args = parser.parse_args()
    create_file(args.output_file, args.accumulators, args.accumulator_size_bits, args.value)


def create_file(output_file, accumulators, accumulator_size_bits, value):
    bitstream = bitarray()
    bitstream += bin(int(value))[2:].zfill(int(accumulator_size_bits)) * int(accumulators)
    bitstream += '0' * (8 - len(bitstream))

    with open(output_file, "wb") as file:
        bitstream.tofile(file)

if __name__ == "__main__":
    main()
