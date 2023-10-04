from src import header as hd

def main():

    header = hd.Header()
    print(header.user_defined_data)
    print(header.sample_type.name)
    print(header.sample_type.value)
    print(header.large_d_flag.name)
    print(header.large_d_flag.value)
    print("Done!")


if __name__ == "__main__":
    main()