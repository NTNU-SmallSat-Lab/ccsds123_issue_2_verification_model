# CCSDS 123.0-B-2 High-level Model

CCSDS 123.0-B-2 High-Level Model is a verification and debugging tool for the development of CCSDS 123.0-B-2 compliant compressors. Its purpose is to input an uncompressed image and output a CCSDS 123.0-B-2 compliant compressed image, along with all intermediate values needed to compress the image.

As this tool is designed for development purposes and to produce a large amount of output data it might be slow and resource-hungry compared to regular CCSDS 123.0-B-2 compression tools. For regular compression, [this CNES-provided tool](https://www.connectbycnes.fr/en/ccsds-1230-b-2-ccsds-1210-b-3) might for example be a more sensible choice.

## Prerequisites  
- Python 3

## How to install and setup this tool

1. Clone or download this repository
2. Install necessary Python packages. Do this by running from the repository root folder (ccsds123_0-b-2_high_level_model): `pip install -r requirements.txt`

## Usage

### Tool overview

To use the tool, run from repo root:

`python ccsds123_0_b_2_high_level_model.py <image_file> [--header HEADER] [--accu ACCU] [--optional OPTIONAL] [--error_limits ERROR_LIMITS]`

Mandatory arguments:
- `image_file`: Path to the raw uncompressed image file. The filename must be in the format `<name>-<datatype>-<z_size>x<y_size>x<x_size>.raw`, as described in the [CCSDS TestData README](https://cwe.ccsds.org/sls/docs/SLS-DC/123.0-B-Info/TestData/README.txt). For example Landsat_mountain-u16be-6x50x100.raw.

Optional arguments:
- `--header HEADER`: Path to the CCSDS 123.0-B-2 header binary file used to set compression settings. The header is formatted as it is in a CCSDS 123.0-B-2 compressed image. When no header binary file is provided, the configuration set in the properties of the `Header` class in `/ccsds123_i2_hlm/header.py` is used. The user can change these properties to change the compression configuration.
- `--accu ACCU`: Path to the hybrid encoder accumulator initial values binary file. Stored as unsigned integers in increasing band order, using D+gamma_0 bits, in a file that is zero-padded to the nearest byte at the end.
- `--optional OPTIONAL`: Path to the optional tables binary file. These are tables that could also be stored in the header. Values are stored as they would be in the header. Tables are stored in the order they would be in the header.
- `--error_limits ERROR_LIMITS`: Path to the error limits binary file for when using periodic error limit updating. Values are stored as 16-bit unsigned integers in the same order they would be in the image.

Outputs:

All outputs from the tool are placed in the `/output/` folder. 
- The compressed image is placed in the `/output/z-output-bistream.bin` file.
- Intermediate values are stored in `.csv` files. Refer to the `save_data`-methods of the respective classes in `/ccsds123_i2_hlm/` for the exact ordering of these files.
- The header binary file is placed in the `/output/header.bin` file.
- The standard does not define initial values for the hybrid encoder accumulator or have it encoded in the header. Hence, when the hybrid encoder is used, initial values are placed in the `/output/hybrid_initial_accumulator.bin` file. The file is in the same format as the `ACCU` optional argument file. If not used, the file exists but is empty.
- If header configurations are used where additional information is necessary to decompress the image, and this additional data can be placed in the header, but is not, the additional data is placed in the `/output/optional_tables.bin` file. The file is in the same format as the `OPTIONAL` optional argument file. If not used, the file exists but is empty.
- If periodic error limit updating is used, the error limits are placed in the `/output/error_limits.bin` file. The file is in the same format as the `ERROR_LIMITS` optional argument file. If not used, the file exists but is empty.

### Example: Simple compression of image
To compress an image, run from the repo root folder:

`python ccsds123_0_b_2_high_level_model.py <image_file>`

Concrete example:

`python ccsds123_0_b_2_high_level_model.py raw_images/Landsat_mountain-u16be-6x50x100.raw`

Note:
- Since no header binary file is provided in this example, the configuration set in the properties of the `Header` class in `/ccsds123_i2_hlm/header.py` is used. The user can change these properties to change the compression configuration.

### Example: Compression of image with an external header file

`python ccsds123_0_b_2_high_level_model.py <image_file> --header <header_file>`


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
