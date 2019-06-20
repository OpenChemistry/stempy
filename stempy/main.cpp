#include <iostream>
#include <fstream>

#include "reader.h"
#include "image.h"
#include "equalizer.h"

using namespace std;
using namespace stempy;

int main (int argc, char *argv[])
{
  if (argc != 2) {
    cerr << "Usage: stem <data file path>" << endl;
    return 1;
  }

  // read the data
  stempy::StreamReader reader(argv[1]);

  int count = 0;
  while (true) {
    auto stream = reader.read();

    if (stream.header.version == 0) {
      break;
    }
    // some printing stuff
    cout << "Block count: " << ++count << endl;
    cout << "Version: " << stream.header.version << endl;
    cout << "Images in block: " <<  stream.header.imagesInBlock << endl;
    cout << "Rows: " << stream.header.frameHeight << endl;
    cout << "Columns: " << stream.header.frameWidth << endl;
    cout << "Image numbers: ";
    for (auto n : stream.header.imageNumbers) {
      cout << n << " ";
    }
    cout << endl;

    // generate STEM image
    cout << "Generating STEM images" << endl;
    std::vector<STEMImage> allImages;
    vector<int> innerRadii(stream.header.imagesInBlock, 40);
    vector<int> outerRadii(stream.header.imagesInBlock, 288);
    int width = 160;
    int height = 160;
    // -1 indicate the center of the image
    int centerX = -1;
    int centerY = -1;
    allImages = createSTEMImages(reader.begin(), reader.end(), innerRadii, outerRadii, width, height, centerX, centerY);

    // generate histograms
    cout << "Generating STEM histograms" << endl;
    int numBins = 20;
    std::vector<std::vector<int>> allHistograms = createSTEMHistograms(allImages, 20);

    // pick one index from all the images 
    int index = 10;
    cout << "Image and histogram pair index we are showing is " << index << endl;
    for (int i = 0; i < allHistograms[index].size(); i++)
    {
      cout << allHistograms[index][i] << " ";
    }
    cout << endl;

    // display index STEM image
    // STEMImage myImage = allImages[index];
    // auto data = myImage.data;
    // for (int i = 0; i < width*height; i++)
    // {
    //   cout << data[i] << ", ";
    // }
    // cout << endl;


  }

}
