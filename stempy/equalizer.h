// equalizer
#ifndef equalizer_h
#define equalizer_h

#include "image.h"

#include <vector>

namespace stempy {

// function that computes histogram for all the STEM images
std::vector<int> createSTEMHistogram(STEMImage inImage, const int numBins);
} // namespace stempy

#endif