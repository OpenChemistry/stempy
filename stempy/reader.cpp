#include "reader.h"
#include "image.h"
#include "mask.h"
#include "config.h"
#ifdef SocketIOClientCpp
#include "sioclient.h"
#endif

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ThreadPool.h>

using namespace std;


namespace stempy {

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

  header.scanWidth = headerPositions[index++];
  header.scanHeight = headerPositions[index++];

  // Now get the image numbers
  auto scanXPosition = headerPositions[index];
  auto scanYPosition = headerPositions[index++];
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

void StreamReader::process(int streamId, int concurrency, int width, int height,
    const string& url) {
// SocketIO support
#ifdef SocketIOClientCpp
  SocketIOClient ioClient(url, "stem");
  ioClient.connect();
  auto msg = std::dynamic_pointer_cast<sio::object_message>(sio::object_message::create());
  msg->insert("width", to_string(width));
  msg->insert("height", to_string(height));
  ioClient.emit("stem.size", msg);
#else
  // Silence unused url parameter warning
  (void)url;
#endif

  // Setup threadpool

  // Default to half the number of cores
  if (concurrency == -1) {
    concurrency = std::thread::hardware_concurrency();
  }

  ThreadPool pool(concurrency);
  uint16_t* mask = nullptr;

  std::vector< std::future<vector<STEMValues>> > results;

  while(true) {
    Block b = this->read();

    if (b.header.version == 0) {
      break;
    }

    if (mask == nullptr) {
      mask =
        createAnnularMask(b.header.frameWidth, b.header.frameHeight, 40, 288);
    }

    results.push_back(pool.enqueue([b{ move(b) }, mask]() {
      vector<STEMValues> values;
      for (unsigned i = 0; i < b.header.imagesInBlock; i++) {
        auto data = b.data.get();
        auto imageNumber = b.header.imageNumbers[i];
        auto numberOfPixels = b.header.frameWidth * b.header.frameHeight;
        values.push_back(calculateSTEMValues(
          data, i * numberOfPixels, numberOfPixels, mask, imageNumber));
      }

      return values;
    }));
  }

  int imageId = 1;

#ifdef SocketIOClientCpp
  // We get the first result so we can calculate the number of pixels
  auto firstResult = results[0].get();

  int numberOfPixels = firstResult.size()*results.size();

  auto emitMessage = [&ioClient, &streamId, &imageId, &numberOfPixels](
                       const string& eventName,
                       const std::vector<uint64_t>& pixelValues,
                       const std::vector<uint32_t>& pixelIndexes,
                       int numPixels) {
    auto message = std::dynamic_pointer_cast<sio::object_message>(
      sio::object_message::create());
    message->insert("streamId", to_string(streamId));
    message->insert("imageId", to_string(imageId));
    auto data = std::dynamic_pointer_cast<sio::object_message>(sio::object_message::create());
    // TODO: Can probably get rid these copies
    data->insert("values", std::make_shared<std::string>(
                             reinterpret_cast<const char*>(pixelValues.data()),
                             numPixels * sizeof(uint64_t)));
    data->insert("indexes",
                 std::make_shared<std::string>(
                   reinterpret_cast<const char*>(pixelIndexes.data()),
                   numPixels * sizeof(uint32_t)));
    message->insert("data", data);

    ioClient.emit(eventName, message);
  };

  // Values
  std::vector<uint64_t> pixels(numberOfPixels);
  std::vector<uint32_t> pixelIndexes(numberOfPixels);

  auto i = 0;
  auto processResult = [&pixels, &pixelIndexes,
                        &i](vector<STEMValues>& values) {
    for(auto &value: values) {
      pixels[i] = value.data;
      pixelIndexes[i] = value.imageNumber - 1;
      i++;
    }
  };

  // First process the block we already removed from results to get the number of pixels
  processResult(firstResult);

  // Now iterate over the rest
  for (auto it = std::next(results.begin()); it != results.end(); ++it) {
    auto &&valuesInBlock = *it;
    auto values = valuesInBlock.get();
    processResult(values);
  }

  emitMessage("stem.data", pixels, pixelIndexes, numberOfPixels);
// TODO We can probably remove this block soon.
#else
  int numberOfPixels = width*height;
  uint64_t pixels[numberOfPixels] = { 0 };
  for(auto &&valuesInBlock: results) {
    auto values = valuesInBlock.get();
    for(auto &value: values ) {
      pixels[value.imageNumber - 1] = value.data;
    }
  }
  // Write the partial images to files
  std::ostringstream filename;
  filename << std::setw(3) << std::setfill('0') << streamId << "."
           << std::setw(3) << std::setfill('0') << imageId << ".bin";
  ofstream file(filename.str(), ios::binary);
  file.write(reinterpret_cast<char*>(pixels),
             numberOfPixels * sizeof(uint64_t));
#endif

  delete[] mask;
}

}
