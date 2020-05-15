#include "reader.h"
#include "config.h"
#include "electron.h"
#include "h5cpp/h5readwrite.h"
#include "image.h"
#include "mask.h"

#include <fstream>
#include <future>
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

Header::Header(Dimensions2D frameDimensions_, uint32_t imageNumInBlock_,
               Dimensions2D scanDimensions_, vector<uint32_t>& imageNumbers_)
{
  this->frameDimensions = frameDimensions_;
  this->imagesInBlock = imageNumInBlock_;
  this->scanDimensions = scanDimensions_;
  this->imageNumbers = imageNumbers_;
}

Block::Block(const Header& h)
  : header(h), data(new uint16_t[h.frameDimensions.first *
                                 h.frameDimensions.second * h.imagesInBlock],
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
  header.frameDimensions.second = headerData[index++];
  header.frameDimensions.first = headerData[index++];
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
  header.frameDimensions = FRAME_DIMENSIONS;
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
  header.frameDimensions = FRAME_DIMENSIONS;
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

  header.scanDimensions.first = headerPositions[index++];
  header.scanDimensions.second = headerPositions[index++];

  // Now get the image numbers
  auto scanXPosition = headerPositions[index++];
  auto scanYPosition = headerPositions[index++];
  header.imageNumbers.push_back(scanYPosition * header.scanDimensions.first +
                                scanXPosition);

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

    auto dataSize = b.header.frameDimensions.first *
                    b.header.frameDimensions.second * b.header.imagesInBlock;
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

SectorStreamReader::SectorStreamReader(const vector<string>& files,
                                       uint8_t version)
  : m_files(files), m_version(version)
{
  // Validate version
  switch (m_version) {
    case 4:
    case 5:
      break;
    default:
      std::ostringstream ss;
      ss << "Unsupported version: ";
      ss << m_version;
      throw invalid_argument(ss.str());
  }

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
  if (m_version == 4) {
    header.frameDimensions = SECTOR_DIMENSIONS_VERSION_4;
  } else {
    header.frameDimensions = SECTOR_DIMENSIONS_VERSION_5;
  }
  header.version = m_version;

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

  header.scanDimensions.first = headerPositions[index++];
  header.scanDimensions.second = headerPositions[index++];

  // Now get the image numbers
  auto scanXPosition = headerPositions[index++];
  auto scanYPosition = headerPositions[index++];

  header.imageNumbers.push_back(scanYPosition * header.scanDimensions.first +
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
          frame.block.header.version = m_version;
          frame.block.header.scanNumber = header.scanNumber;
          frame.block.header.scanDimensions = header.scanDimensions;
          frame.block.header.imagesInBlock = 1;
          frame.block.header.imageNumbers.push_back(pos);
          frame.block.header.frameDimensions = FRAME_DIMENSIONS;
          frame.block.data.reset(
            new uint16_t[frame.block.header.frameDimensions.first *
                         frame.block.header.frameDimensions.second],
            std::default_delete<uint16_t[]>());
          std::fill(frame.block.data.get(),
                    frame.block.data.get() +
                      frame.block.header.frameDimensions.first *
                        frame.block.header.frameDimensions.second,
                    0);
        }

        readSectorData(frame.block, sector);
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

void SectorStreamReader::readSectorData(std::ifstream& stream, Block& block,
                                        int sector)
{
  if (version() == 4) {
    readSectorDataVersion4(stream, block, sector);
  } else {
    readSectorDataVersion5(stream, block, sector);
  }
}

void SectorStreamReader::readSectorData(Block& block, int sector)
{
  auto& sectorStream = *m_streamsIterator;
  auto& stream = sectorStream.stream;

  readSectorData(*stream.get(), block, sector);
}

void SectorStreamReader::readSectorDataVersion4(std::ifstream& stream,
                                                Block& block, int sector)
{
  auto frameX = sector * SECTOR_DIMENSIONS_VERSION_4.first;
  for (unsigned frameY = 0; frameY < FRAME_DIMENSIONS.second; frameY++) {
    auto offset = FRAME_DIMENSIONS.first * frameY + frameX;
    read(stream, block.data.get() + offset,
         SECTOR_DIMENSIONS_VERSION_4.first * sizeof(uint16_t));
  }
}

void SectorStreamReader::readSectorDataVersion5(std::ifstream& stream,
                                                Block& block, int sector)
{
  auto frameY = sector * SECTOR_DIMENSIONS_VERSION_5.second;
  auto offset = frameY * FRAME_DIMENSIONS.first;

  read(stream, block.data.get() + offset,
       SECTOR_DIMENSIONS_VERSION_5.first * SECTOR_DIMENSIONS_VERSION_5.second *
         sizeof(uint16_t));
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
        auto dataSize = header.frameDimensions.first *
                        header.frameDimensions.second * header.imagesInBlock;
        stream.seekg(dataSize * sizeof(uint16_t), stream.cur);
      };

      auto block = [&stream, &header]() -> Block {
        Block b(header);
        auto dataSize = header.frameDimensions.first *
                        header.frameDimensions.second * header.imagesInBlock;
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
  Dimensions2D scanDimensions;

  auto func = [&numberOfSectors, &scanDimensions](int sector, Header& header,
                                                  auto& skip, auto& block) {
    (void)block;
    (void)sector;
    numberOfSectors++;
    scanDimensions = header.scanDimensions;
    skip();
  };

  readAll(func);

  auto expectedNumberOfSectors =
    scanDimensions.first * scanDimensions.second * 4;

  return static_cast<float>(numberOfSectors) / expectedNumberOfSectors;
}

void SectorStreamReader::toHdf5FrameFormat(h5::H5ReadWrite& writer)
{
  bool created = false;
  std::vector<int> dims;

  for (auto iter = this->begin(); iter != this->end(); ++iter) {
    auto b = std::move(*iter);

    // When we receive the first header we can create the file
    if (!created) {
      dims.push_back(static_cast<int>(b.header.scanDimensions.first) *
                     static_cast<int>(b.header.scanDimensions.second));
      dims.push_back(static_cast<int>(FRAME_DIMENSIONS.first));
      dims.push_back(static_cast<int>(FRAME_DIMENSIONS.first));

      std::vector<int> chunkDims = {
        1, static_cast<int>(FRAME_DIMENSIONS.first),
        static_cast<int>(FRAME_DIMENSIONS.second)
      };
      writer.createDataSet("/", "frames", dims,
                           h5::H5ReadWrite::DataType::UInt16, chunkDims);
      std::vector<int> scanSize = {
        0, static_cast<int>(b.header.scanDimensions.second),
        static_cast<int>(b.header.scanDimensions.first)
      };
      writer.createGroup("/stem");
      writer.createDataSet("/stem", "images", scanSize,
                           h5::H5ReadWrite::DataType::UInt64);
      created = true;
    }

    size_t start[3] = { 0, 0, 0 };
    size_t counts[3] = { 1, b.header.frameDimensions.second,
                         b.header.frameDimensions.first };
    for (unsigned i = 0; i < b.header.imagesInBlock; i++) {
      auto pos = b.header.imageNumbers[i];
      auto offset = i * FRAME_DIMENSIONS.first * FRAME_DIMENSIONS.second;
      start[0] = pos;

      auto data = b.data.get() + offset;
      if (!writer.updateData("/frames", dims, h5::H5ReadWrite::DataType::UInt16,
                             data, start, counts)) {
        throw std::runtime_error("Unable to update HDF5.");
      }
    }
  }
}

void SectorStreamReader::toHdf5DataCubeFormat(h5::H5ReadWrite& writer)
{
  bool created = false;
  std::vector<int> dims;

  for (auto iter = this->begin(); iter != this->end(); ++iter) {
    auto b = std::move(*iter);

    // When we receive the first header we can create the file
    if (!created) {
      dims.push_back(static_cast<int>(b.header.scanDimensions.first));
      dims.push_back(static_cast<int>(b.header.scanDimensions.second));
      dims.push_back(static_cast<int>(FRAME_DIMENSIONS.first));
      dims.push_back(static_cast<int>(FRAME_DIMENSIONS.first));

      std::vector<int> chunkDims = {
        1, 1, static_cast<int>(FRAME_DIMENSIONS.first),
        static_cast<int>(FRAME_DIMENSIONS.second)
      };
      writer.createDataSet("/", "datacube", dims,
                           h5::H5ReadWrite::DataType::UInt16, chunkDims);
      created = true;
    }

    size_t start[4] = { 0, 0, 0, 0 };
    size_t counts[4] = { 1, 1, b.header.frameDimensions.second,
                         b.header.frameDimensions.first };
    for (unsigned i = 0; i < b.header.imagesInBlock; i++) {
      auto pos = b.header.imageNumbers[0];
      auto offset = i * FRAME_DIMENSIONS.first * FRAME_DIMENSIONS.second;
      auto x = pos % b.header.scanDimensions.first;
      auto y = pos / b.header.scanDimensions.first;
      start[0] = x;
      start[1] = y;

      auto data = b.data.get() + offset;
      if (!writer.updateData("/datacube", dims,
                             h5::H5ReadWrite::DataType::UInt16, data, start,
                             counts)) {
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

bool operator==(const SectorStreamReader::SectorStream& lhs,
                const SectorStreamReader::SectorStream& rhs)
{
  return lhs.stream.get() == rhs.stream.get();
}

SectorStreamThreadedReader::SectorStreamThreadedReader(const std::string& path,
                                                       uint8_t version)
  : SectorStreamReader(path, version)
{
  initNumberOfThreads();
}

SectorStreamThreadedReader::SectorStreamThreadedReader(
  const std::vector<std::string>& files, uint8_t version)
  : SectorStreamReader(files, version)
{
  initNumberOfThreads();
}

SectorStreamThreadedReader::SectorStreamThreadedReader(const std::string& path,
                                                       uint8_t version,
                                                       int threads)
  : SectorStreamReader(path, version), m_threads(threads)
{
  initNumberOfThreads();
}

SectorStreamThreadedReader::SectorStreamThreadedReader(
  const std::vector<std::string>& files, uint8_t version, int threads)
  : SectorStreamReader(files, version), m_threads(threads)
{
  initNumberOfThreads();
}

void SectorStreamThreadedReader::initNumberOfThreads()
{
  if (m_threads < 1) {
    m_threads = std::thread::hardware_concurrency();
    // May not be able to determine
    if (m_threads == 0) {
      std::cout << "WARNING: Unable to determine hardware concurrency, "
                   "defaulting to 10."
                << std::endl;
      m_threads = 10;
    }
  }
}

bool SectorStreamThreadedReader::nextSectorStreamPair(
  SectorStreamPair& sectorStreamPair)
{
  {
    std::unique_lock<std::mutex> queueLock(m_queueMutex);
    if (m_streamQueue.empty()) {
      return false;
    }
    sectorStreamPair = m_streamQueue.front();
    m_streamQueue.pop();
  }

  auto& stream = sectorStreamPair.first;
  auto sector = sectorStreamPair.second;

  auto c = stream->peek();
  // If we have reached the end close the stream and remove if from
  // the list.
  if (c == EOF) {
    stream->close();
    auto iter = m_streams.begin();
    while (iter != m_streams.end()) {
      if ((*iter).stream.get() == stream) {
        break;
      }
      iter++;
    }

    m_streams.erase(iter);

    return false;
  }

  return true;
}
}
