#include <iostream>
#include <fstream>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include "reader.h"
#include "image.h"
#include "equalizer.h"

using namespace std;
using namespace stempy;
namespace fs = boost::filesystem;

// Helper function that checks if given string path is of a Directory
bool checkIfDirectory(std::string filePath)
{
	try 
  {
    // Create a Path object from given path string
    fs::path pathObj(filePath);
    // Check if path exists and is of a directory file
    if (fs::exists(pathObj) && fs::is_directory(pathObj))
      return true;
	}
	catch (fs::filesystem_error & e)
	{
		std::cerr << e.what() << std::endl;
	}
	return false;
}

// helper function that gets all the file names in a directory
vector<string> GetDirectoryFiles(const string& dir) 
{
  vector<string> allFiles;
  // Create a Path object from given path string
  fs::path pathObj(dir);

  fs::directory_iterator end_iter;
  for ( fs::directory_iterator dir_itr(pathObj); dir_itr != end_iter; ++dir_itr )
  {
    try
    {
      if ( fs::is_regular_file( dir_itr->status() ) )
      {
        allFiles.push_back(dir_itr->path().filename().string());
      }
    }
    catch ( const std::exception & ex )
    {
      std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
    }
  }

  return allFiles;
}

int main (int argc, char *argv[])
{
  if (argc != 2) {
    cerr << "Usage: stem <data file path>" << endl;
    return 1;
  }

  string inputPath = argv[1];

  // check the input path
  if (checkIfDirectory(inputPath))
  {
    cout << endl << "Input path " << inputPath << " is valid, starting processing" << endl;
    // get all the files in this path
    const vector<string> files = GetDirectoryFiles(inputPath);
    int numFiles = files.size();
    cout << numFiles << " files have been found" << endl;

    // all the STEM images that would be summed into one final STEM image
    vector<STEMImage> allPartialSTEMImages;

    // information about processing
    int width = 160;
    int height = 160;
    // -1 indicate the center of the image
    int centerX = -1;
    int centerY = -1;

    for (int i = 0; i < numFiles; i++)
    {
      cout << "Processing " << i+1 << "/" << numFiles << ": " << files[i] << endl;
      string curFile = files[i];
      // read the data
      stempy::StreamReader reader(inputPath+curFile);
      auto stream = reader.read();

      // version number needs to be 1, 2 or 3
      if (stream.header.version == 0) 
      {
        break;
      }
      // some printing stuff about the input data
      cout << "Version: " << stream.header.version << endl;
      cout << "Images in block: " <<  stream.header.imagesInBlock << endl;
      cout << "Rows: " << stream.header.frameHeight << endl;
      cout << "Columns: " << stream.header.frameWidth << endl;
      cout << "Diffraction image numbers: ";
      for (auto n : stream.header.imageNumbers) {
        cout << n << " ";
      }
      cout << endl;

      // generate current partial STEM image
      cout << "Generating partial STEM image" << endl;
      
      // information about how to process
      vector<int> innerRadii(stream.header.imagesInBlock, 40);
      vector<int> outerRadii(stream.header.imagesInBlock, 288);
      
      // create STEM 32 partial STEM images
      vector<STEMImage> imagesOfBlock = createSTEMImages(reader.begin(), reader.end(), innerRadii, outerRadii, width, height, centerX, centerY);

      // sum up all the partial images
      STEMImage curPartialImage = imagesOfBlock[0];
      for (int j = 1; j < imagesOfBlock.size(); j++)
      {
        for (int k = 0; k < width*height; k++)
          curPartialImage.data[k] = curPartialImage.data[k] + imagesOfBlock[j].data[k];
      }

      allPartialSTEMImages.push_back(curPartialImage);

    }

    // summing all the partial images from each block
    STEMImage finalSTEMImage = allPartialSTEMImages[0];
    for (int i = 1; i < allPartialSTEMImages.size(); i++)
    {
      for (int k = 0; k < width*height; k++)
        finalSTEMImage.data[k] = finalSTEMImage.data[k] + allPartialSTEMImages[i].data[k];
    }

    // print the final STEM image
    for (int i = 0; i < width*height; i++)
    {
      cout << finalSTEMImage.data[i] << ", ";
    }
    cout << endl;

    // generate histograms
    cout << "Generating STEM histograms" << endl;
    int numBins = 100;
    std::vector<int> histogram = createSTEMHistograms(finalSTEMImage, numBins);

    // print the generated histogram
    cout << "Histogram of current STEM image is " << endl;
    for (int i = 0; i < histogram.size(); i++)
    {
      cout << histogram[i] << " ";
    }
    cout << endl;

  }

}
