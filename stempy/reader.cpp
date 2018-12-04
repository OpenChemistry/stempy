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
    return m_stream.read(reinterpret_cast<char*>(value), size);
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

Stream StreamReader::read() {
  Stream s;
  s.header = readHeader();

  auto dataSize = s.header.rows*s.header.columns*s.header.imagesInBlock;
  s.data.reset(new uint16_t[dataSize]);
  read(s.data.get(), dataSize*sizeof(uint16_t));

  return s;
}

}
