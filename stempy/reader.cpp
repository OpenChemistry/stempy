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

Block::Block(const Header& header) :
  header(header),
  data(new uint16_t[header.rows*header.columns*header.imagesInBlock],
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
  header.rows = headerData[index++];
  header.columns = headerData[index++];
  header.version = headerData[index++];
  header.timestamp =  headerData[index++];
  // Skip over 6 - 10 - reserved
  index += 5;

  // Now get the image numbers
  header.imageNumbers.resize(header.imagesInBlock);
  copy(headerData + index,
       headerData + index + header.imagesInBlock,
       header.imageNumbers.data());

  return header;
}

Header StreamReader::readHeaderVersion2() {

  Header header;

  uint32_t firstImageNumber;
  read(&firstImageNumber, sizeof(uint32_t));
  // HACK!
  // Our current datasets doesn't seem to have a valid firstImageNumber, so we
  // reset to zero here!
  firstImageNumber = 1;

  header.imagesInBlock = 1600;
  header.rows = 576;
  header.columns = 576;
  header.version = 2;

  // Now generate the image numbers
  header.imageNumbers.reserve(header.imagesInBlock);
  for(int i=0; i<header.imagesInBlock; i++) {
    header.imageNumbers.push_back(firstImageNumber + i);
  }

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
      default:
        std::ostringstream ss;
        ss << "Unexpected version: ";
        ss << this->m_version;
        throw invalid_argument(ss.str());
    }

    Block b(header);

    auto dataSize = b.header.rows * b.header.columns * b.header.imagesInBlock;
    read(b.data.get(), dataSize * sizeof(uint16_t));

    return b;
  } catch (EofException& e) {
    throw invalid_argument("Unexpected EOF while processing stream.");
  }

  return Block();
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
#endif

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

    if (brightFieldMask == nullptr) {
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

  int imageId = 1;

#ifdef SocketIOClientCpp
  // We get the first result so we can calculate the number of pixels
  auto firstResult = results[0].get();

  int numberOfPixels = firstResult.size()*results.size();

  auto emitMessage = [&ioClient, &streamId, &imageId, &numberOfPixels](
    const string& eventName, const std::vector<uint64_t>& pixelValues,
    const std::vector<uint32_t>& pixelIndexes, int numberOfPixes) {
    auto msg = std::dynamic_pointer_cast<sio::object_message>(sio::object_message::create());
    msg->insert("streamId", to_string(streamId));
    msg->insert("imageId", to_string(imageId));
    auto data = std::dynamic_pointer_cast<sio::object_message>(sio::object_message::create());
    // TODO: Can probably get rid these copies
    data->insert("values", std::make_shared<std::string>(
                             reinterpret_cast<const char*>(pixelValues.data()),
                             numberOfPixels * sizeof(uint64_t)));
    data->insert("indexes",
                 std::make_shared<std::string>(
                   reinterpret_cast<const char*>(pixelIndexes.data()),
                   numberOfPixels * sizeof(uint32_t)));
    msg->insert("data", data);

    ioClient.emit(eventName, msg);
  };

  // Values
  std::vector<uint64_t> brightPixels(numberOfPixels);
  std::vector<uint64_t> darkPixels(numberOfPixels);
  std::vector<uint32_t> pixelIndexes(numberOfPixels);

  auto i = 0;
  auto processResult = [&brightPixels, &darkPixels, &pixelIndexes, &i](vector<STEMValues> &values) {
    for(auto &value: values) {
      brightPixels[i] = value.bright;
      darkPixels[i] = value.dark;
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

  emitMessage("stem.bright", brightPixels, pixelIndexes, numberOfPixels);
  emitMessage("stem.dark", darkPixels, pixelIndexes, numberOfPixels);
// TODO We can probably remove this block soon.
#else
  int numberOfPixels = width*height;
  uint64_t brightPixels[numberOfPixels]  = {0};
  uint64_t darkPixels[numberOfPixels]  = {0};
  for(auto &&valuesInBlock: results) {
    auto values = valuesInBlock.get();
    for(auto &value: values ) {
      brightPixels[value.imageNumber-1] = value.bright;
      darkPixels[value.imageNumber-1] = value.dark;
    }
  }
  // Write the partial images to files
  std::ostringstream brightFilename, darkFilename;
  brightFilename << "bright-" << std::setw( 3 ) <<  std::setfill('0')  <<
      streamId << "."  << std::setw( 3 ) <<  std::setfill('0')  << imageId << ".bin";
  darkFilename << "dark-" << std::setw( 3 ) <<  std::setfill('0')  << streamId
      << "." << std::setw( 3 ) <<  std::setfill('0')  << imageId << ".bin";
  ofstream brightFile(brightFilename.str(), ios::binary);
  ofstream darkFile(darkFilename.str(), ios::binary);
  brightFile.write(reinterpret_cast<char*>(brightPixels), numberOfPixels*sizeof(uint64_t));
  darkFile.write(reinterpret_cast<char*>(darkPixels), numberOfPixels*sizeof(uint64_t));
#endif

  delete[] brightFieldMask;
  delete[] darkFieldMask;
}

}
