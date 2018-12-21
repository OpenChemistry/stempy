#include "reader.h"
#include "image.h"
#include "mask.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ThreadPool.h>

using namespace std;


namespace stempy {

Block::Block(const Header& header) :
  header(header),
  data(new uint16_t[header.rows*header.columns*header.imagesInBlock],
      std::default_delete<uint16_t[]>())
{}

StreamReader::StreamReader(const std::string& path)
{
  m_stream.open (path, ios::in | ios::binary);
  if (!m_stream.is_open()) {
    ostringstream msg;
    msg << "Unable to open file: " << path;
    throw invalid_argument(msg.str());
  }
}

template<typename T>
istream & StreamReader::read(T& value){
    return read(&value, sizeof(value));
}

template<typename T>
istream & StreamReader::read(T* value, streamsize size){
    m_stream.read(reinterpret_cast<char*>(value), size);

    if (m_stream.eof()) {
      throw EofException();
    }

    return m_stream;
}

Header StreamReader::readHeader() {

  Header header;

  uint32_t headerData[1024];
  read(headerData, 1024*sizeof(uint32_t));

  int index = 0;
  header.imagesInBlock = headerData[index++];
  header.rows = headerData[index++];
  header.columns = headerData[index++];
  header.version = headerData[index++];
  header.timestamp =  headerData[index++];
  // Skip over 6 - 10 - reserved
  index += 5;

  // Now get the image numbers
  header.imageNumbers.resize(header.imagesInBlock);
  auto imageNumbersSize = sizeof(uint32_t)*header.imagesInBlock;
  copy(headerData + index,
       headerData + index + header.imagesInBlock,
       header.imageNumbers.data());

  return header;
}

Block StreamReader::read() {


  // Check that we have a block to read
  auto c = m_stream.peek();
  if (c != EOF) {
    try {
      auto header = readHeader();
      Block b(header);

      auto dataSize = b.header.rows*b.header.columns*b.header.imagesInBlock;
      read(b.data.get(), dataSize*sizeof(uint16_t));

      return b;
    }
    catch (EofException& e) {
      throw invalid_argument("Unexpected EOF while processing stream.");
    }
  }
  return Block();
}

void StreamReader::process(int streamId, int concurrency, int width, int height) {
  // Setup threadpool

  // Default to half the number of cores
  if (concurrency == -1) {
    concurrency = std::thread::hardware_concurrency();
  }

  ThreadPool pool(concurrency);
  uint16_t* brightFieldMask = nullptr;
  uint16_t* darkFieldMask = nullptr;

  std::vector< std::future<vector<STEMValues>> > results;

  while(true) {
    Block b = this->read();

    if (b.header.version == 0) {
      break;
    }

    if (brightFieldMask == nullptr) {40,
      brightFieldMask = createAnnularMask(b.header.rows, b.header.columns, 0, 288);
    }

    if (darkFieldMask == nullptr) {
      darkFieldMask = createAnnularMask(b.header.rows, b.header.columns, 40, 288);
    }

    results.push_back(pool.enqueue([b{move(b)}, brightFieldMask, darkFieldMask]() {
      vector<STEMValues> values;
      for (int i=0; i<b.header.imagesInBlock; i++) {
        auto data = b.data.get();
        auto imageNumber = b.header.imageNumbers[i];
        auto numberOfPixels = b.header.rows*b.header.columns;
        values.push_back(calculateSTEMValues(data, i*numberOfPixels,
            numberOfPixels, brightFieldMask, darkFieldMask, imageNumber));
      }

      return values;
    }));
  }


  uint64_t brightPixels[width*height]  = {0};
  uint64_t darkPixels[width*height]  = {0};
  for(auto &&valuesInBlock: results) {
    auto values = valuesInBlock.get();
    for(auto &value: values ) {
      brightPixels[value.imageNumber-1] = value.bright;
      darkPixels[value.imageNumber-1] = value.dark;
    }
  }

  int imageId = 1;

  // Write the partial images to files
  std::ostringstream brightFilename, darkFilename;
  brightFilename << "bright-" << std::setw( 3 ) <<  std::setfill('0')  <<
      streamId << "."  << std::setw( 3 ) <<  std::setfill('0')  << imageId << ".bin";
  darkFilename << "dark-" << std::setw( 3 ) <<  std::setfill('0')  << streamId
      << "." << std::setw( 3 ) <<  std::setfill('0')  << imageId << ".bin";
  ofstream brightFile(brightFilename.str(), ios::binary);
  ofstream darkFile(darkFilename.str(), ios::binary);
  brightFile.write(reinterpret_cast<char*>(brightPixels), width*height*sizeof(uint64_t));
  darkFile.write(reinterpret_cast<char*>(darkPixels), width*height*sizeof(uint64_t));
}

}
