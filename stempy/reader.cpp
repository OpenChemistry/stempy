#include "reader.h"
#include "image.h"
#include "mask.h"
#include "config.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ThreadPool.h>

using std::copy;
using std::invalid_argument;
using std::ios;
using std::istream;
using std::move;
using std::ofstream;
using std::ostringstream;
using std::streamsize;
using std::string;
using std::vector;

namespace stempy {

Header::Header(uint32_t frameWidth_, uint32_t frameHeight_,
               uint32_t imageNumInBlock_, uint32_t scanWidth_,
               uint32_t scanHeight_, vector<uint32_t>& imageNumbers_)
{
    this->frameWidth = frameWidth_;
    this->frameHeight = frameHeight_;
    this->imagesInBlock = imageNumInBlock_;
    this->scanHeight = scanHeight_;
    this->scanWidth = scanWidth_;
    this->imageNumbers = imageNumbers_;
}

Block::Block(const Header& h)
  : header(h),
    data(new uint16_t[h.frameWidth * h.frameHeight * h.imagesInBlock],
         std::default_delete<uint16_t[]>())
{}

StreamReader::StreamReader(const vector<string>& files, uint8_t version)
  : m_files(files), m_version(version)
{
  // If there are no files, throw an exception
  if (m_files.empty()) {
    ostringstream msg;
    msg << "No files provided to StreamReader!";
    throw invalid_argument(msg.str());
  }

  // Open up the first file
  openNextFile();
}

void StreamReader::openNextFile()
{
  // If we already have a file open, move on to the next index
  // Otherwise, assume we haven't opened any files yet
  if (m_stream.is_open()) {
    m_stream.close();
    ++m_curFileIndex;
  }

  // Don't do anything if we are at the end of the files
  if (atEnd())
    return;

  const auto& file = m_files[m_curFileIndex];
  m_stream.open(file, ios::in | ios::binary);

  if (!m_stream.is_open()) {
    ostringstream msg;
    msg << "Unable to open file: " << file;
    throw invalid_argument(msg.str());
  }
}

template<typename T>
istream & StreamReader::read(T& value){
    return read(&value, sizeof(value));
}

template<typename T>
istream & StreamReader::read(T* value, streamsize size){

  if (atEnd())
    throw EofException();

  m_stream.read(reinterpret_cast<char*>(value), size);

  // If we are at the end of the file, open up the next file
  // and try again.
  if (m_stream.eof()) {
    openNextFile();
    return read(value, size);
  }

  return m_stream;
}

Header StreamReader::readHeaderVersion1() {

  Header header;

  uint32_t headerData[1024];
  read(headerData, 1024*sizeof(uint32_t));

  int index = 0;
  header.imagesInBlock = headerData[index++];
  header.frameHeight = headerData[index++];
  header.frameWidth = headerData[index++];
  header.version = headerData[index++];
  header.timestamp =  headerData[index++];
  // Skip over 6 - 10 - reserved
  index += 5;

  // Now get the image numbers
  header.imageNumbers.resize(header.imagesInBlock);
  copy(headerData + index,
       headerData + index + header.imagesInBlock,
       header.imageNumbers.data());

  // Currently the imageNumbers seem to be 1 indexed, we hope this will change.
  // for now, convert them to 0 indexed to make the rest of the code easier.
  auto& imageNumbers = header.imageNumbers;
  for (unsigned i = 0; i < header.imagesInBlock; i++) {
    imageNumbers[i]-= 1;
  }

  return header;
}

Header StreamReader::readHeaderVersion2() {

  Header header;

  uint32_t firstImageNumber;
  read(&firstImageNumber, sizeof(uint32_t));
  // HACK!
  // Our current datasets doesn't seem to have a valid firstImageNumber, so we
  // reset to zero here!
  firstImageNumber = 0;

  header.imagesInBlock = 1600;
  header.frameWidth = 576;
  header.frameHeight = 576;
  header.version = 2;

  // Now generate the image numbers
  header.imageNumbers.reserve(header.imagesInBlock);
  for (unsigned i = 0; i < header.imagesInBlock; i++) {
    header.imageNumbers.push_back(firstImageNumber + i);
  }

  return header;
}

// This was the documented format, but the current firmware implementation
// has the x y order reversed.
// unsigned int32 scan_number;
// unsigned int32 frame_number;
// unsigned int16 total_number_of_stem_x_positions_in_scan;
// unsigned int16 total_number_of_stem_y_positions_in_scan;
// unsigned int16 stem_x_position_of_frame;
// unsigned int16 stem_y_position_of_frame;
Header StreamReader::readHeaderVersion3()
{

  Header header;

  header.imagesInBlock = 1;
  header.frameWidth = 576;
  header.frameHeight = 576;
  header.version = 3;

  // Read scan and frame number
  uint32_t headerNumbers[2];
  read(headerNumbers, 2 * sizeof(uint32_t));

  int index = 0;
  header.scanNumber = headerNumbers[index++];
  header.frameNumber = headerNumbers[index];

  // Now read the size and positions
  uint16_t headerPositions[4];
  index = 0;
  read(headerPositions, 4 * sizeof(uint16_t));

  // Note: The order is currently reversed y then x rather than the other way around
  header.scanHeight = headerPositions[index++];
  header.scanWidth = headerPositions[index++];

  // Now get the image numbers
  auto scanYPosition = headerPositions[index++];
  auto scanXPosition = headerPositions[index++];
  header.imageNumbers.push_back(scanYPosition * header.scanWidth  + scanXPosition);

  return header;
}

Block StreamReader::read()
{
  if (atEnd())
    return Block();

  // Check that we have a block to read
  auto c = m_stream.peek();

  // If we are at the end of the file, open up the next file and try
  // again
  if (c == EOF) {
    openNextFile();
    return read();
  }

  try {
    Header header;
    switch (this->m_version) {
      case 1:
        header = readHeaderVersion1();
        break;
      case 2:
        header = readHeaderVersion2();
        break;
      case 3:
        header = readHeaderVersion3();
        break;
      default:
        std::ostringstream ss;
        ss << "Unexpected version: ";
        ss << this->m_version;
        throw invalid_argument(ss.str());
    }

    Block b(header);

    auto dataSize =
      b.header.frameWidth * b.header.frameHeight * b.header.imagesInBlock;
    read(b.data.get(), dataSize * sizeof(uint16_t));

    return b;
  } catch (EofException& e) {
    throw invalid_argument("Unexpected EOF while processing stream.");
  }

  return Block();
}

void StreamReader::reset()
{
  if (m_stream.is_open())
    m_stream.close();

  m_curFileIndex = 0;
}

}
