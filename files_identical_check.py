import argparse

def main():
    parser = argparse.ArgumentParser(description="Check if the contents of two files are identical.")
    parser.add_argument("file1", help="Path to the first file")
    parser.add_argument("file2", help="Path to the second file")
    args = parser.parse_args()

    with open(args.file1, 'rb') as file1, open(args.file2, 'rb') as file2:
        content1 = file1.read()
        content2 = file2.read()

    if content1 == content2:
        print("Files are identical")
    else:
        print("Files are different")


if __name__ == "__main__":
    main()
