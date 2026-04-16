# CCSDS 123.0-B-2 High-level Model

CCSDS 123.0-B-2 High-Level Model is a verification and debugging tool for the development of CCSDS 123.0-B-2 compliant compressors. Its purpose is to input an uncompressed image and output a CCSDS 123.0-B-2 compliant compressed image, along with all intermediate values needed to compress the image. Furthermore, to allow modifications to the algorithm outside of the CCSDS specification, support for decompression of compressed image files is also supported.

As this tool is designed for development purposes and produces a large amount of output data, it might be slow and resource-hungry compared to regular CCSDS 123.0-B-2 compression tools. For regular compression and decompression within CCSDS specification, [this CNES-provided tool](https://www.connectbycnes.fr/en/ccsds-1230-b-2-ccsds-1210-b-3) might for example be a more sensible choice. To slightly remedy the calculation speed, the predictor module is written in C++ using a library named _pybind11_ to allow integration with Python.

## Usage Rights

This repository is licensed under the MIT License.

If you use this code in your research, please cite our paper:

D. Vorhaug, S. Boyle and M. Orlandić, "High-Level CCSDS 123.0-B-2 Hyperspectral Image Compressor Verification Model", Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS), Helsinki, Finland, Dec. 2024

> As of June 2026, the codebase has been modified to include decompression.

## Prerequisites  
- Python 3
- Cmake
- A C++ compiler

## Install package

Clone the repository and run `pip install .` from the repository root directory.

## Setup for development

1. Clone or download this repository
2. Install necessary Python packages. Do this by running from the repository root folder (ccsds123_0-b-2_high_level_model): `pip install -r requirements.txt`
3. Compile C++ predictor with the following commands:
```
> mkdir build
> cd build
> cmake .. -Dpybind11_DIR=/path-to-pybind11
> make install
```
This installs the C++ predictor sub-module as a shared object file into the _ccsds123_i2_hlm_ folder. If changes are made to the predictor, rebuild it using `make install`.

> _pybind11_DIR_ needs to be set to the path of the _pybind11Config.cmake_ corresponding to your installation. When using _pip_ as above, this path can be found by looking at the _Location_ field in `pip show pybind11`, and then appending `/pybind11/share/cmake/pybind11`.

## Usage

### Tool overview and examples

The tool can be used as a command line tool, or integrated into other python projects. Some simple examples will be shown here, but for a full explanation of command line flags run the help command as follows.

`python ccsds123_0_b_2_high_level_model.py --help`

Two main operations are provided, compression and decompression, controlled by the presence of the `--decompress` flag.

Raw image files should be formatted as `<name>-<datatype>-<z_size>x<y_size>x<x_size>.raw`, and may be compressed using:
`python ccsds123_0_b_2_high_level_model.py raw_images/Landsat_mountain-u16be-6x50x100.raw --header raw_images/landsat-hdr.bin`

To decompress an image run:
`python ccsds123_0_b_2_high_level_model.py output/z-output-bistream-dec.bin --file_format u16be --decompress`

If no header file is provided for the compressor the header config will use the defaults from the `Header` class in `/ccsds123_i2_hlm/header.py`. Files specifying hybrid encoder initial accumulator values, header optional values, and error limit tables for periodic error limit updates may also be provided when applicable.

When decompressing, the header config is read from the compressed bitstream. Furthermore, the file format of the decompressed image file must be provided on the command line.

Outputs:

All outputs from the tool are placed in the `/output/` folder. Some additional intermediate results from the predictor may be stored by adding the command line flag `--save_intermediates`. 
- The compressed image bitstream is placed in the `/output/z-output-bistream-enc.bin` file.
- Decompressed image is placed in the `/output/z-output-bistream-dec.bin` file.
- Intermediate values are stored in `.csv` files. Refer to the `save_data`-methods of the respective classes in `/ccsds123_i2_hlm/` for the exact ordering of these files.
- The header binary file is placed in the `/output/header.bin` file.
- The standard does not define initial values for the hybrid encoder accumulator or have it encoded in the header. Hence, when the hybrid encoder is used, initial values are placed in the `/output/hybrid_initial_accumulator.bin` file. The file is in the same format as the `ACCU` optional argument file. If not used, the file exists but is empty.
- If header configurations are used where additional information is necessary to decompress the image, and this additional data can be placed in the header, but is not, the additional data is placed in the `/output/optional_tables.bin` file. The file is in the same format as the `OPTIONAL` optional argument file. If not used, the file exists but is empty.
- If periodic error limit updating is used, the error limits are placed in the `/output/error_limits.bin` file. The file is in the same format as the `ERROR_LIMITS` optional argument file. If not used, the file exists but is empty.

> When decompressing, only the resulting image is stored. This can be changed by adding the _save_data_ methods manually in the code, but might overwrite values from the compressor when using the verification script.

## Verification

The CCSDS 123.0-B-2 High-Level Model is verified by testing against the CCSDS provided test vector set `Test1-20190201` available from [TestVectors-B2](https://cwe.ccsds.org/sls/docs/Forms/AllItems.aspx?RootFolder=%2fsls%2fdocs%2fsls%2ddc%2f123%2e0%2dB%2dInfo%2fTestData%2fTestVectors%2dB2&FolderCTID=0x012000439B56FF51847E41B5728F9730D7B55F). Users can do this for themselves, to gain confidence in the tool or to verify changes they have done themselves, by downloading and extracting the set `Test1-20190201` and running:

`python verify.py <test_vector_folder_path>`

The included Makefile can also be used to verify the model against the trusted CNES-provided tool. The tool directory has to be added to PATH for this to be used. To compare the model to the CNES tool with the model built-in header configuration run from the terminal:

`make compare image=<image_file>`

To compare with optional files, run:

`make compare_with_optionals image=<image_file> image_format=<image_format> header=<header_file> optional_tables=<optional_tables_file> error_limits=<error_limits_file> accu=<hybrid_initial_accumulator_file>`

Point to empty files if some optional files are not used.
Concrete example:

`make compare_with_optionals image=raw_images/Landsat_mountain-u16be-6x50x100.raw image_format=u16be header=header.bin optional_tables=optional_tables.bin error_limits=error_limits.bin accu=accu.bin`


## Other useful notes
- Hyperspectral test images can be found on [CCSDS TestData](https://cwe.ccsds.org/sls/docs/Forms/AllItems.aspx?RootFolder=%2Fsls%2Fdocs%2FSLS%2DDC%2F123%2E0%2DB%2DInfo%2FTestData). 
- To speed up the compression of test images, the `tools/crop_image.py` tool can be used to crop images smaller. 
