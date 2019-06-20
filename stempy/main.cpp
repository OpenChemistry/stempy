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
    // vector<STEMImage> allPartialSTEMImage;

    for (int i = 0; i < numFiles; i++)
    {
      cout << files[i] << endl;
      // vector<STEMImage> curPartialSTEMImage = 
    }
/*
    // read the data
    stempy::StreamReader reader(argv[1]);

    int count = 0;
    while (true) 
    {
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
    }

    // display index STEM image
    // STEMImage myImage = allImages[index];
    // auto data = myImage.data;
    // for (int i = 0; i < width*height; i++)
    // {
    //   cout << data[i] << ", ";
    // }
    // cout << endl;
*/

  }

}
