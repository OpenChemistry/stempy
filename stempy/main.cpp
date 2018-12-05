#include <iostream>
#include <fstream>

#include "reader.h"

using namespace std;

int main (int argc, char *argv[])
{
  if (argc != 2) {
    cerr << "Usage: stem <data file path>" << endl;
    return 1;
  }

  stempy::StreamReader reader(argv[1]);

  int count = 0;
  while (true) {
    auto stream = reader.read();

    if (stream.header.version == 0) {
      break;
    }
    cout << "Block count: " << ++count << endl;
    cout << "Version: " << stream.header.version << endl;
    cout << "Images in block: " <<  stream.header.imagesInBlock << endl;
    cout << "Rows: " <<  stream.header.rows << endl;
    cout << "Columns: " <<  stream.header.columns << endl;
    cout << "Image numbers: ";

    for (auto n : stream.header.imageNumbers) {
      cout << n << " ";
    }
    cout << endl;

  }

}
