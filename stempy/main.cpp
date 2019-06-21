#include <fstream>
#include <iostream>
#include <string>
// #include <experimental/filesystem>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include "equalizer.h"
#include "image.h"
#include "reader.h"

using namespace std;
using namespace stempy;
namespace fs = boost::filesystem;
// namespace fs = std::experimental::filesystem;

// Helper function that checks if given string path is of a Directory
bool checkIfDirectory(std::string filePath)
{
  try {
    // Create a Path object from given path string
    fs::path pathObj(filePath);
    // Check if path exists and is of a directory file
    if (fs::exists(pathObj) && fs::is_directory(pathObj))
      return true;
  } catch (fs::filesystem_error& e) {
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
  for (fs::directory_iterator dir_itr(pathObj); dir_itr != end_iter;
       ++dir_itr) {
    try {
      if (fs::is_regular_file(dir_itr->status())) {
        allFiles.push_back(dir_itr->path().filename().string());
      }
    } catch (const std::exception& ex) {
      std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
    }
  }
  // for (const auto& entry : fs::directory_iterator(pathObj))
  // {
  //   auto filename = entry.path().filename();
  //   if (fs::is_regular_file(entry.status()))
  //   {
  //     allFiles.push_back(filename);
  //   }
  // }

  return allFiles;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    cerr << "Usage: stem <data file path>" << endl;
    return 1;
  }

  string inputPath = argv[1];

  // check the input path
  if (checkIfDirectory(inputPath)) {
    cout << endl
         << "Input path " << inputPath << " is valid, starting processing"
         << endl;
    // get all the files in this path
    const vector<string> files = GetDirectoryFiles(inputPath);
    int numFiles = files.size();
    cout << numFiles << " files have been found" << endl;

    // all the STEM images that would be summed into one final STEM image
    vector<vector<STEMImage>> allPartialSTEMImages;

    // information about processing
    int width = 160;
    int height = 160;
    // -1 indicate the center of the image
    int centerX = -1;
    int centerY = -1;
    int numRadii = 1;

    for (int i = 0; i < numFiles; i++) {
      cout << "Processing " << i + 1 << "/" << numFiles << ": " << files[i]
           << endl;
      string curFile = files[i];
      // read the data
      stempy::StreamReader reader(inputPath + curFile);
      auto stream = reader.read();

      // version number needs to be 1, 2 or 3
      if (stream.header.version == 0) {
        break;
      }
      // some printing stuff about the input data
      cout << "Version: " << stream.header.version << endl;
      cout << "Images in block: " << stream.header.imagesInBlock << endl;
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
      vector<int> innerRadii(numRadii, 40);
      vector<int> outerRadii(numRadii, 288);
      // sanity check
      if (innerRadii.size() != numRadii)
      {
        cerr << "radii vector has different size than numRadii" << endl;
      }

      // create numRadii STEM images
      vector<STEMImage> curPartialImage =
        createSTEMImages(reader.begin(), reader.end(), innerRadii, outerRadii,
                         width, height, centerX, centerY);

      // for later sum up
      allPartialSTEMImages.push_back(curPartialImage);
    }

    // sum up based on different radii
    vector<STEMImage> finalSTEMImages;
    for (int i = 0; i < numRadii; i++) {
      STEMImage singleRadiiSTEMImage = allPartialSTEMImages[i][0];
      // sum up based on different blocks
      for (int j = 1; j < allPartialSTEMImages[i].size(); j++)
      {
        for (int k = 0; k < width * height; k++)
        {
          singleRadiiSTEMImage.data[k] = singleRadiiSTEMImage.data[k] + allPartialSTEMImages[i][0].data[k];
        }
      }
      // push into final results
      finalSTEMImages.push_back(singleRadiiSTEMImage);
    }

    cout << finalSTEMImages.size() 
         << " STEM image(s) has been generated corresponding to " 
         << numRadii << " mask radius" << endl;

    // print the first radii STEM image
    // for (int i = 0; i < width * height; i++) {
    //   cout << finalSTEMImages[0].data[i] << ", ";
    // }
    // cout << endl;

    // generate histograms
    int numBins = 500;
    cout << endl << "Generating STEM histograms with " << numBins << " bins" << endl;
    vector<vector<int>> allRadiiHistograms;
    for (int i = 0; i < numRadii; i++)
    {
      std::vector<int> singleRadiiHistogram = createSTEMHistogram(finalSTEMImages[i], numBins);
      allRadiiHistograms.push_back(singleRadiiHistogram);
    }
    

    // print the generated histogram
    cout << "Histogram corresponding to radii index 0 STEM image is " << endl;
    for (int i = 0; i < allRadiiHistograms[0].size(); i++) {
      cout << allRadiiHistograms[0][i] << " ";
    }
    cout << endl;
  }
}
