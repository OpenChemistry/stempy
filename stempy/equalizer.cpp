// contrast equalizer file
#include "equalizer.h"

// todo: parallel computation
#ifdef VTKm
#endif

#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>

namespace stempy {
// hide helper functions
namespace {
// helper function that returns the binding result
std::vector<float> getContainer(const STEMImage inImage, const int numBins)
{
  // find the min and max value across all the input STEM images
  auto min = std::numeric_limits<float>::max();
  auto max = std::numeric_limits<float>::min();

  // information about input STEMImage
  int width = inImage.width;
  int height = inImage.height;
  auto curData = inImage.data;
  std::cout << "Input STEM Image has width = " << width
            << ", height = " << height << std::endl;

  // curImage.data is shared_ptr<uint64_t []>
  for (int i = 0; i < width * height; i++) {
    // std::cout << "curData[" << i << "] = " << curData[i] << std::endl;
    if (curData[i] < min)
      min = curData[i];
    if (curData[i] > max)
      max = curData[i];
  }

  // the "length" of each slot of the container
  float length = (float)((max - min) / numBins);

  std::vector<float> container;
  // push the min value first
  container.push_back(min);
  // push all the intermediate values
  for (int i = 1; i < numBins; i++) {
    container.push_back(i * length);
  }
  // push the max value as the last number
  container.push_back(max);

  // print the container binders
  // std::cout << "Container is " << std::endl;
  // for (int i = 0; i < container.size(); i++)
  // {
  //     std::cout << container[i] << " ";
  // }
  // std::cout << std::endl;

  return container;
}
} // namespace

// function that computes histogram for all the STEM images
// each histogram is a vector<int>
std::vector<int> createSTEMHistograms(STEMImage inImage, const int numBins)
{
  // get the container of the histogram
  std::vector<float> container = getContainer(inImage, numBins);

  // initialize output
  std::vector<std::vector<int>> allHistrograms;

  // iterate through all the input STEM image
  int width = inImage.width;
  int height = inImage.height;
  auto curData = inImage.data;
  // for each image, get a histrogram
  std::vector<int> histrogram(numBins, 0);
  for (int i = 0; i < width * height; i++) {
    auto value = curData[i];
    // check which bin it belongs to
    for (int i = 0; i < numBins; i++) {
      if (value >= container[i] && value < container[i + 1]) {
        histrogram[i] += 1;
      }
      // the max value is put in the last slot
      else if (value == container[numBins]) {
        histrogram[numBins - 1] += 1;
      }
    }
  }

  return histrogram;
}
} // namespace stempy