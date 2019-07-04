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

# helper function that generates and saves histogram image
def save_hist(i, bins, freq, out_dir):
    num_bins = len(bins) - 1
    bins = [str(element) for element in bins]
    # plt.bar considers the left boundary
    x = np.arange(num_bins + 1)
    # init figure
    fig = plt.figure(i, figsize=(16, 8))
    my_hist = fig.add_subplot(111)
    my_hist.bar(x[:-1], freq[0], align='edge')
    plt.xticks(x[::20], bins[::20])
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # save to local
    plt.savefig(out_dir)

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

    # generate STEM image
    imgs = image._create_stem_images(reader, inner_radii, outer_radii,
                                    width=160, height=160)
    imgs_np = [np.array(img, copy=False) for img in imgs]
    imgs_np = np.array(imgs_np, copy=False)

    # perform equalization
    num_bins = 100
    num_hist = 1
    imgs_equalized = image.standard_histogram_equalization(imgs, num_bins)
    imgs_equalized_np = [np.array(img, copy=False) for img in imgs_equalized]
    imgs_equalized_np = np.array(imgs_equalized_np, copy=False)

    # save both raw and equalized STEM images
    for i, img_np in enumerate(imgs_np):
        suffix = str(inner_radii[i]) + '_' + str(outer_radii[i]) + '.png'
        save_img(img_np, out_dir + '/img_' + suffix)
    for i, img_equalized_np in enumerate(imgs_equalized_np):
        suffix = str(inner_radii[i]) + '_' + str(outer_radii[i]) + '.png'
        save_img(img_equalized_np, out_dir + '/equalized_img_' + suffix)

    # generate histogram of unequalized image for verification
    all_bins_uneq, all_freqs_uneq = image.histograms_from_images(imgs, num_bins, num_hist)
    suffix_uneq = str(inner_radii[i]) + '_' + str(outer_radii[i]) + '.png'
    save_path_uneq = out_dir + '/histogram_' + suffix_uneq
    save_hist(0, all_bins_uneq[0], all_freqs_uneq[0], save_path_uneq)
    print('histogram_' + suffix_uneq + ' has been saved')

    # generate histogram of equalized image for verification
    all_bins_eq, all_freqs_eq = image.histograms_from_images(imgs_equalized, num_bins, num_hist)
    suffix_eq = 'equalized_' + str(inner_radii[i]) + '_' + str(outer_radii[i]) + '.png'
    save_path_eq = out_dir + '/histogram_' + suffix_eq
    save_hist(1, all_bins_eq[0], all_freqs_eq[0], save_path_eq)
    print('histogram_' + suffix_eq + ' has been saved')


if __name__ == "__main__":
    main(sys.argv[1:])
