import sys
import os
from stempy import io, image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def print_help():
    print(
    '''
    Usage:
    python3 stem_histogram.py [data_dir -- path to the data directory]
                              [out_dir -- path to the output image directory]
    ''')

def save_img(stem_image_data, name):
    min = np.min(stem_image_data)
    max = np.max(stem_image_data)

    stem_image_data = stem_image_data.reshape((160, 160))
    stem_image_data = np.interp(stem_image_data, [min, max], [0, 256])
    stem_image_data = stem_image_data.astype(np.uint8)
    img = Image.fromarray(stem_image_data)
    img.save(name)

def main(argv):
    print_help()
    data_dir = argv[0]
    out_dir = argv[1]

    # create output directory if it does not exit
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('Output directory', out_dir , 'created')
    else:
        print('Output directory', out_dir , 'already exists')

    # get all the data files
    files = []
    for root, dirs, fs in os.walk(data_dir):
        for f in fs:
            files.append(os.path.join(root, f))

    # inner and outer radius of mask
    inner_radii = [40]
    outer_radii = [288]

    # file reader
    reader = io.reader(files)

    # generate histograms
    num_bins = 100
    # how many sub-histograms in one image
    num_hist = 11
    print('Generating histograms for input data')
    all_bins, all_freqs = image.create_stem_histogram(num_bins, num_hist, reader,
                                                      inner_radii, outer_radii,
                                                      width=160, height=160)

    # plot histogram
    for i in range(len(all_bins)):
        # obtain current bins and freq
        bins = [str(element) for element in all_bins[i]]
        freq = all_freqs[i]
        # plt.bar considers the left boundary
        x = np.arange(num_bins+1)
        for j in range(num_hist):
            # init figure
            fig = plt.figure(j, figsize=(16, 8))
            my_hist = fig.add_subplot(111)
            my_hist.bar(x[:-1], freq[j], align='edge')
            plt.xticks(x[::20], bins[::20])
            if (j == 1):
                plt.title(str(j) + 'st histogram of STEM image with inner radius = '
                        + str(inner_radii[i]) + ', outer radius = ' + str(outer_radii[i]))
            elif (j == 2):
                plt.title(str(j) + 'nd histogram of STEM image with inner radius = '
                        + str(inner_radii[i]) + ', outer radius = ' + str(outer_radii[i]))
            else:
                plt.title(str(j) + 'th histogram of STEM image with inner radius = '
                        + str(inner_radii[i]) + ', outer radius = ' + str(outer_radii[i]))
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            # save to local
            suffix = str(j) + '_' + str(inner_radii[i]) + '_' + str(outer_radii[i]) + '.png'
            plt.savefig(out_dir + '/histogram_' + suffix)
            print('histogram_' + suffix + ' has been saved')

            # plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
