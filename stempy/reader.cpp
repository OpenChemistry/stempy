#include "reader.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;


namespace stempy {

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

  read(header.imagesInBlock);
  read(header.rows);
  read(header.columns);
  read(header.version);
  read(header.timestamp);
  // Skip over 6 - 10 - reserved
  m_stream.seekg(5*sizeof(uint32_t), m_stream.cur);

  // Now get the image numbers
  header.imageNumbers.resize(header.imagesInBlock);
  read(header.imageNumbers.data(), sizeof(uint32_t)*header.imagesInBlock);

  return header;
}

Block StreamReader::read() {
  Block b;

  // Check that we have a block to read
  auto c = m_stream.peek();
  if (c != EOF) {
    try {
      b.header = readHeader();

      auto dataSize = b.header.rows*b.header.columns*b.header.imagesInBlock;
      b.data.reset(new uint16_t[dataSize]);
      read(b.data.get(), dataSize*sizeof(uint16_t));
    }
    catch (EofException& e) {
      throw invalid_argument("Unexpected EOF while processing stream.");
    }
  }

  return b;
}

}
