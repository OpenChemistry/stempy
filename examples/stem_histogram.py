import glob
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
    python3 stem_histogram.py [dataDir -- path to the data directory]
                              [outDir -- path to the output image directory]
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
    dataDir = argv[0] if len(argv) > 0 else '/home/zhuokai/Desktop/data/data*.dat'
    outDir = argv[1] if len(argv) > 1 else '/home/zhuokai/Desktop/stempy/examples/output/'

    # create output directory if it does not exit
    if not os.path.exists(outDir):
        os.mkdir(outDir)
        print('Output directory', outDir , 'created')
    else:    
        print('Outout directory', outDir , 'already exists')

    # get the all the data files
    files = []
    for f in glob.glob(dataDir):
        files.append(f)

    # inner and outer radius of mask
    inner_radii = [40]
    outer_radii = [288]

    reader = io.reader(files)
    imgs = image.create_stem_images(reader, inner_radii, outer_radii, width=160,
                                    height=160)

    # save the STEM images
    for i, img in enumerate(imgs):
        # convert the type to numpy array
        img_np = np.array(img, copy=False)
        suffix = str(inner_radii[i]) + '_' + str(outer_radii[i]) + '.png'
        save_img(img_np, outDir + '/img_' + suffix)
        print('STEM image with inner radius = ' + str(inner_radii[i]) 
              + ', outer radius = ' + str(outer_radii[i]) + ' has been saved')

    # generate histograms
    numBins = 100
    for i, img in enumerate(imgs):
        print('Generating histogram for STEM image ' + str(i))
        bins, freq = image.create_stem_histogram(img, numBins)
        bins = [str(element) for element in bins]

        # plot histogram
        fig = plt.figure(1, figsize=(16, 8))
        myHist = fig.add_subplot(121)
        # plt.bar considers the left boundary
        x = np.arange(numBins+1)
        myHist.bar(x[:-1], freq, align='edge')
        plt.xticks(x[::20], bins[::20])
        plt.title('My histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # histogram using built-in python histogram
        npHist = fig.add_subplot(122)
        npHist.hist(img, normed=0)
        plt.title('Numpy histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # save to local
        suffix = str(inner_radii[i]) + '_' + str(outer_radii[i]) + '.png'
        plt.savefig(outDir + '/histogram_' + suffix)
        
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])

    