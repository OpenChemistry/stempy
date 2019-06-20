// equalizer
#ifndef equalizer_h
#define equalizer_h

#include "image.h"

#include <vector>

namespace stempy {

    // function that computes histogram for all the STEM images
    std::vector<std::vector<int>> createSTEMHistograms (std::vector<STEMImage> allImages, const int numBins);
}

#endif