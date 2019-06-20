// contrast equalizer file
#include "equalizer.h"

// todo: parallel computation
#ifdef VTKm
#endif

#include <cmath>
#include <future>
#include <algorithm>
#include <iostream>


namespace stempy 
{
    // hide helper functions
    namespace 
    {
        // helper function that returns the binding result
        std::vector<float> getcontainer(const std::vector<STEMImage> allImages, const int numBins)
        {
            // find the min and max value across all the input STEM images
            auto min = std::numeric_limits<float>::max();
            auto max = std::numeric_limits<float>::min();

            // STEMImage is Image<uint64_t>
            for (STEMImage curImage : allImages)
            {
                auto curMin = std::numeric_limits<float>::max();
                auto curMax = -std::numeric_limits<float>::max();
                int width = curImage.width;
                int height = curImage.height;
                auto curData = curImage.data;
                std::cout << "Current STEM Image has width = " << width << ", height = " << height << std::endl;
                std::cout << "curMin is " << curMin << std::endl;
                std::cout << "curMax is " << curMax << std::endl;
                // curImage.data is shared_ptr<uint64_t []>
                for (int i = 0; i < width*height; i++)
                {
                    // std::cout << "curData[" << i << "] = " << curData[i] << std::endl;
                    if (curData[i] < curMin)
                        curMin = curData[i];
                    if (curData[i] > curMax)
                        curMax = curData[i];
                }
                // update
                if (curMin < min)
                    min = curMin;
                if (curMax > max)
                    max = curMax;
            }

            std::cout << "Min is " << min << std::endl;
            std::cout << "Max is " << max << std::endl;

            // the "length" of each slot of the container
            float length = (float)((max - min) / numBins);

            std::vector<float> container;
            // push the min value first
            container.push_back(min);
            // push all the intermediate values
            for (int i = 1; i < numBins; i++)
            {
                container.push_back(i*length);
            }
            // push the max value as the last number
            container.push_back(max);

            // print the container binders
            std::cout << "Container is " << std::endl;
            for (int i = 0; i < container.size(); i++)
            {
                std::cout << container[i] << " ";
            }
            std::cout << std::endl;

            return container;
        }
    }
    
    // function that computes histogram for all the STEM images
    // each histogram is a vector<int>
    std::vector<std::vector<int>> createSTEMHistograms (const std::vector<STEMImage> allImages, const int numBins)
    {
        // number of input images
        int numImage = allImages.size();

        // get the container of the histogram
        std::vector<float> container = getcontainer(allImages, numBins);

        // initialize output
        std::vector<std::vector<int>> allHistrograms;
        
        // iterate through all the input STEM image
        for (STEMImage curImage : allImages)
        {
            int width = curImage.width;
            int height = curImage.height;
            auto curData = curImage.data;
            // for each image, get a histrogram
            std::vector<int> curHistogram(numBins, 0);
            for (int i = 0; i < width*height; i++)
            {
                auto value = curData[i];
                // check which bin it belongs to
                for (int i = 0; i < numBins; i++)
                {
                    if (value >= container[i] && value < container[i+1])
                    {
                        curHistogram[i] += 1;
                    }
                    // the max value is put in the last slot
                    else if (value == container[numBins])
                    {
                        curHistogram[numBins-1] += 1;
                    }
                    else
                    {
                        // std::cout << "Unexpected error happens when generating histogram" << std::endl;
                    }
                    
                }

                // push
                allHistrograms.push_back(curHistogram);
            }
        }

        return allHistrograms;
    }
}