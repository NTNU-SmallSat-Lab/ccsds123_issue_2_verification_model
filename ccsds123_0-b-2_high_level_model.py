from src import header as hd

imageName = 'aviris_maine_f030828t01p00r05_sc10.uncal-u16be-224x512x680.raw'
imageName = 'Landsat_mountain-u16be-6x1024x1024.raw'

def main():

    header = hd.Header(imageName)
    # print(header.user_defined_data)
    # print(header.sample_type.name)
    # print(header.sample_type.value)
    # print(header.large_d_flag.name)
    # print(header.large_d_flag.value)

    print(header.x_size)
    print(header.y_size)
    print(header.z_size)
    print(header.sample_type.name)
    print(header.large_d_flag.name)
    print(header.dynamic_range)











    print("Done!")


if __name__ == "__main__":
    main()