#include "reader.h"
#include "config.h"
#include "h5cpp/h5readwrite.h"
#include "image.h"
#include "mask.h"

#include <ThreadPool.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <vector>

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

const int SECTOR_WIDTH = 144;
const int SECTOR_HEIGHT = 576;
const int FRAME_WIDTH = 576;
const int FRAME_HEIGHT = 576;

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

bool StreamReader::openNextFile()
{
  // If we already have a file open, move on to the next index
  // Otherwise, assume we haven't opened any files yet
  if (m_stream.is_open()) {
    m_stream.close();
    ++m_curFileIndex;
  }

  // Don't do anything if we are at the end of the files
  if (atEnd())
    return false;

  const auto& file = m_files[m_curFileIndex];
  m_stream.open(file, ios::in | ios::binary);

  if (!m_stream.is_open()) {
    ostringstream msg;
    msg << "Unable to open file: " << file;
    throw invalid_argument(msg.str());
  }

  return true;
}

istream& StreamReader::skip(std::streamoff offset)
{
  m_stream.seekg(offset, m_stream.cur);

  return m_stream;
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
  header.frameWidth = FRAME_WIDTH;
  header.frameHeight = FRAME_HEIGHT;
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
  header.frameWidth = FRAME_WIDTH;
  header.frameHeight = FRAME_HEIGHT;
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

// SectorStreamReader

int extractSector(const std::string& fileName)
{
  std::regex sectorRegex(".*module(\\d+).*\\.data");
  std::smatch matches;
  if (std::regex_search(fileName, matches, sectorRegex)) {
    return std::stoi(matches[1]);
  }

  return -1;
}

SectorStreamReader::SectorStreamReader(const vector<string>& files)
  : m_files(files)
{
  // If there are no files, throw an exception
  if (m_files.empty()) {
    ostringstream msg;
    msg << "No files provided to SectorStreamReader!";
    throw invalid_argument(msg.str());
  }

  openFiles();
  m_streamsIterator = m_streams.begin();
}

SectorStreamReader::~SectorStreamReader()
{
  m_streams.clear();
}

// This was the documented format, but the current firmware implementation
// has the x y order reversed.
// unsigned int32 scan_number;
// unsigned int32 frame_number;
// unsigned int16 total_number_of_stem_x_positions_in_scan;
// unsigned int16 total_number_of_stem_y_positions_in_scan;
// unsigned int16 stem_x_position_of_frame;
// unsigned int16 stem_y_position_of_frame;
Header SectorStreamReader::readHeader(std::ifstream& stream)
{

  Header header;

  header.imagesInBlock = 1;
  header.frameWidth = SECTOR_WIDTH;
  header.frameHeight = SECTOR_HEIGHT;
  header.version = 4;

  // Read scan and frame number
  uint32_t headerNumbers[2];
  read(stream, headerNumbers, 2 * sizeof(uint32_t));

  int index = 0;
  header.scanNumber = headerNumbers[index++];
  header.frameNumber = headerNumbers[index];

  // Now read the size and positions
  uint16_t headerPositions[4];
  index = 0;
  read(stream, headerPositions, 4 * sizeof(uint16_t));

  header.scanHeight = headerPositions[index++];
  header.scanWidth = headerPositions[index++];

  // Now get the image numbers
  auto scanYPosition = headerPositions[index++];
  auto scanXPosition = headerPositions[index++];

  header.imageNumbers.push_back(scanYPosition * header.scanWidth +
                                scanXPosition);

  return header;
}

Header SectorStreamReader::readHeader()
{
  auto& sectorStream = *m_streamsIterator;
  auto& stream = sectorStream.stream;

  return readHeader(*stream.get());
}

istream& SectorStreamReader::skip(std::streamoff offset)
{
  auto& sectorStream = *m_streamsIterator;
  auto& stream = sectorStream.stream;
  stream->seekg(offset, stream->cur);

  return *stream;
}

template <typename T>
istream& SectorStreamReader::read(T& value)
{
  return read(&value, sizeof(value));
}

template <typename T>
istream& SectorStreamReader::read(T* value, streamsize size)
{
  if (atEnd())
    throw EofException();

  auto& sectorStream = *m_streamsIterator;
  auto& stream = sectorStream.stream;

  return stream->read(reinterpret_cast<char*>(value), size);
}

template <typename T>
std::istream& SectorStreamReader::read(std::ifstream& stream, T& value)
{
  return read(stream, &value, sizeof(value));
}

template <typename T>
std::istream& SectorStreamReader::read(std::ifstream& stream, T* value,
                                       std::streamsize size)
{
  return stream.read(reinterpret_cast<char*>(value), size);
}

Block SectorStreamReader::read()
{
  while (!m_streams.empty()) {
    while (m_streamsIterator != m_streams.end()) {
      auto& sectorStream = *m_streamsIterator;
      auto& stream = sectorStream.stream;
      auto sector = sectorStream.sector;
      auto c = stream->peek();
      // If we have reached the end close the stream and remove if from
      // the list.
      if (c == EOF) {
        stream->close();
        m_streamsIterator = m_streams.erase(m_streamsIterator);
        continue;
      }

      auto header = readHeader();
      for (unsigned i = 0; i < header.imagesInBlock; i++) {
        auto pos = header.imageNumbers[i];
        auto& frame = m_frameCache[pos];

        // Do we need to allocate the frame
        if (frame.block.header.version == 0) {
          frame.block.header.version = 4;
          frame.block.header.scanNumber = header.scanNumber;
          frame.block.header.scanWidth = header.scanWidth;
          frame.block.header.scanHeight = header.scanHeight;
          frame.block.header.imagesInBlock = 1;
          frame.block.header.imageNumbers.push_back(pos);
          frame.block.header.frameWidth = FRAME_WIDTH;
          frame.block.header.frameHeight = FRAME_HEIGHT;
          frame.block.data.reset(new uint16_t[frame.block.header.frameWidth *
                                              frame.block.header.frameHeight],
                                 std::default_delete<uint16_t[]>());
          std::fill(
            frame.block.data.get(),
            frame.block.data.get() + frame.block.header.frameWidth * frame.block.header.frameHeight, 0);
        }

        auto frameX = sector * SECTOR_WIDTH;
        for (unsigned frameY = 0; frameY < FRAME_HEIGHT; frameY++) {
          auto offset = FRAME_WIDTH * frameY + frameX;
          read(frame.block.data.get() + offset,
               SECTOR_WIDTH * sizeof(uint16_t));
        }
        frame.sectorCount++;

        if (frame.sectorCount == 4) {
          auto b = frame.block;
          m_frameCache.erase(pos);
          m_streamsIterator++;

          return b;
        }
      }

      m_streamsIterator++;
    }
    // Start iterating from the beginning
    if (!m_streams.empty()) {
      m_streamsIterator = m_streams.begin();
    }
  }

  // Now  return the partial frames
  if (!m_frameCache.empty()) {
    auto iter = m_frameCache.begin();
    auto& frame = (*iter).second;
    auto block = frame.block;
    m_frameCache.erase(iter);

    return block;
  }

  return Block();
}

void SectorStreamReader::openFiles()
{

  for (auto& file : m_files) {
    auto stream = new std::ifstream();
    stream->open(file, ios::in | ios::binary);

    if (!stream->is_open()) {
      delete stream;
      ostringstream msg;
      msg << "Unable to open file: " << file;
      throw invalid_argument(msg.str());
    }

    auto sector = extractSector(file);

    m_streams.emplace_back(stream, sector);
  }
}

void SectorStreamReader::reset()
{
  for (auto& s : m_streams) {
    if (s.stream->is_open()) {
      s.stream->close();
    }
  }
  m_streams.clear();
  openFiles();
  m_streamsIterator = m_streams.begin();
}

template <typename Functor>
void SectorStreamReader::readAll(Functor func)
{
  for (auto& file : m_files) {
    std::ifstream stream;
    stream.open(file, ios::in | ios::binary);
    if (!stream.is_open()) {
      ostringstream msg;
      msg << "Unable to open file: " << file;
      throw invalid_argument(msg.str());
    }

    auto sector = extractSector(file);

    while (true) {
      // Check if we are done
      auto c = stream.peek();
      if (c == EOF) {
        stream.close();
        break;
      }
      auto header = readHeader(stream);
      auto skip = [&stream, &header]() {
        auto dataSize =
          header.frameWidth * header.frameHeight * header.imagesInBlock;
        stream.seekg(dataSize * sizeof(uint16_t), stream.cur);
      };

      auto block = [&stream, &header]() -> Block {
        Block b(header);
        auto dataSize =
          header.frameWidth * header.frameHeight * header.imagesInBlock;
        stream.read(reinterpret_cast<char*>(b.data.get()),
                    dataSize * sizeof(uint16_t));

        return b;
      };

      func(sector, header, skip, block);
    }
  }
}

float SectorStreamReader::dataCaptured()
{
  uint64_t numberOfSectors = 0;
  uint32_t scanWidth;
  uint32_t scanHeight;

  auto func = [&numberOfSectors, &scanWidth, &scanHeight](
                int sector, Header& header, auto& skip, auto& block) {
    (void)block;
    (void)sector;
    numberOfSectors++;
    scanWidth = header.scanWidth;
    scanHeight = header.scanHeight;
    skip();
  };

  readAll(func);

  auto expectedNumberOfSectors = scanWidth * scanHeight * 4;

  return static_cast<float>(numberOfSectors) / expectedNumberOfSectors;
}

void SectorStreamReader::toHdf5FrameFormat(h5::H5ReadWrite& writer)
{
  bool created = false;
  for (auto iter = this->begin(); iter != this->end(); ++iter) {
    auto b = std::move(*iter);

    // When we receive the first header we can create the file
    if (!created) {
      std::vector<int> dims = { b.header.scanWidth * b.header.scanHeight,
                                FRAME_WIDTH, FRAME_WIDTH };
      std::vector<int> chunkDims = { 1, FRAME_WIDTH, FRAME_HEIGHT };
      writer.createDataSet("/", "frames", dims,
                           h5::H5ReadWrite::DataType::UInt16, chunkDims);
      std::vector<int> scanSize = { 0, b.header.scanHeight, b.header.scanWidth };
      writer.createGroup("/stem");
      writer.createDataSet("/stem", "images", scanSize,
                           h5::H5ReadWrite::DataType::UInt64);
      created = true;
    }

    size_t start[3] = { 0, 0, 0 };
    size_t counts[3] = { 1, b.header.frameHeight, b.header.frameWidth };
    for (unsigned i = 0; i < b.header.imagesInBlock; i++) {
      auto pos = b.header.imageNumbers[0];
      auto offset = i * FRAME_WIDTH * FRAME_HEIGHT;
      start[0] = pos;

      auto data = b.data.get() + offset;
      if (!writer.updateData("/frames", h5::H5ReadWrite::DataType::UInt16, data,
                             start, counts)) {
        throw std::runtime_error("Unable to update HDF5.");
      }
    }
  }
}

void SectorStreamReader::toHdf5DataCubeFormat(h5::H5ReadWrite& writer)
{
  bool created = false;
  for (auto iter = this->begin(); iter != this->end(); ++iter) {
    auto b = std::move(*iter);

    // When we receive the first header we can create the file
    if (!created) {
      std::vector<int> dims = { b.header.scanWidth, b.header.scanHeight,
                                FRAME_WIDTH, FRAME_WIDTH };
      std::vector<int> chunkDims = { 1, 1, FRAME_WIDTH, FRAME_HEIGHT };
      writer.createDataSet("/", "datacube", dims,
                           h5::H5ReadWrite::DataType::UInt16, chunkDims);
      created = true;
    }

    size_t start[4] = { 0, 0, 0, 0 };
    size_t counts[4] = { 1, 1, b.header.frameHeight, b.header.frameWidth };
    for (unsigned i = 0; i < b.header.imagesInBlock; i++) {
      auto pos = b.header.imageNumbers[0];
      auto offset = i * FRAME_WIDTH * FRAME_HEIGHT;
      auto x = pos % b.header.scanWidth;
      auto y = pos / b.header.scanWidth;
      start[0] = x;
      start[1] = y;

      auto data = b.data.get() + offset;
      if (!writer.updateData("/datacube", h5::H5ReadWrite::DataType::UInt16,
                             data, start, counts)) {
        throw std::runtime_error("Unable to update HDF5.");
      }
    }
  }
}

void SectorStreamReader::toHdf5(const std::string& path,
                                SectorStreamReader::H5Format format)
{
  h5::H5ReadWrite::OpenMode mode = h5::H5ReadWrite::OpenMode::WriteOnly;
  h5::H5ReadWrite writer(path.c_str(), mode);

  if (format == H5Format::Frame) {
    toHdf5FrameFormat(writer);
  } else {
    toHdf5DataCubeFormat(writer);
  }
}
}
