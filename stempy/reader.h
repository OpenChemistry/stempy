#ifndef stempyreader_h
#define stempyreader_h

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <chrono>

using Clock = std::chrono::system_clock;

namespace h5 {
class H5ReadWrite;
}

namespace stempy {

// Convention is (x, y)
using Coordinates2D = std::pair<int, int>;

// Convention is (width, height)
using Dimensions2D = std::pair<uint32_t, uint32_t>;

struct EofException : public std::exception
{
  const char* what () const throw () { return "EOF Exception"; }
};

struct Header {
  Dimensions2D scanDimensions = { 0, 0 };
  Dimensions2D frameDimensions = { 0, 0 };
  uint32_t imagesInBlock = 0, version = 0, timestamp = 0;
  uint32_t frameNumber = 0, scanNumber = 0;
  std::vector<uint32_t> imageNumbers;

  Header() = default;
  Header(const Header& header) = default;
  Header(Header&& header) noexcept = default;
  Header& operator=(Header&& header) noexcept = default;
  Header(Dimensions2D frameDimensions, uint32_t imageNumInBlock,
         Dimensions2D scanDimensions, std::vector<uint32_t>& imageNumbers);
};

struct Block {
  Header header;
  std::shared_ptr<uint16_t> data;

  Block() = default;
  Block(const Block&) = default;
  Block(const Header& header);
  Block(Block&& i) noexcept = default;
  Block& operator=(Block&& i) noexcept = default;
};

template <typename T>
class BlockIterator
{
public:
  using self_type = BlockIterator;
  using value_type = Block;
  using reference = Block&;
  using pointer = Block*;
  using iterator_category = std::input_iterator_tag;
  using difference_type = void; // Differences not allowed here

  // This class is required for "*it++" to work properly
  // But we really shouldn't use "*it++", because it is expensive
  class postinc_return
  {
  public:
    postinc_return(reference value_) : value(value_) {}
    value_type operator*() { return value; }

  private:
    value_type value;
  };

  BlockIterator(T* reader) : m_streamReader(reader)
  {
    if (reader)
      ++(*this);
  }

  self_type operator++()
  {
    m_block = m_streamReader->read();
    if (!m_block.data)
      m_streamReader = nullptr;
    return *this;
  }

  postinc_return operator++(int)
  {
    postinc_return temp(m_block);
    ++(*this);
    return temp;
  }

  reference operator*() { return m_block; }

  pointer operator->() { return &m_block; }

  bool operator==(const self_type& rhs)
  {
    return m_streamReader == rhs.m_streamReader;
  }

  bool operator!=(const self_type& rhs) { return !(*this == rhs); }

private:
  T* m_streamReader;
  value_type m_block;
};

class StreamReader
{

public:
  StreamReader(const std::string& path, uint8_t version = 1);
  StreamReader(const std::vector<std::string>& files, uint8_t version = 1);

  Block read();

  // Reset to the start of the first file
  void reset();

  typedef BlockIterator<StreamReader> iterator;
  iterator begin() { return iterator(this); }
  iterator end() { return iterator(nullptr); }

private:
  std::ifstream m_stream;
  std::vector<std::string> m_files;
  size_t m_curFileIndex = 0;
  int m_version;
  int m_sector = -1;

  bool openNextFile();

  // Whether or not we are at the end of all of the files
  bool atEnd() const { return m_curFileIndex >= m_files.size(); }

  Header readHeaderVersion1();
  Header readHeaderVersion2();
  Header readHeaderVersion3();

  template<typename T>
  std::istream & read(T& value);
  template<typename T>
  std::istream & read(T* value, std::streamsize size);
  std::istream& skip(std::streamoff pos);
  int sector() { return m_sector; };
};

inline StreamReader::StreamReader(const std::string& path, uint8_t version)
  : StreamReader(std::vector<std::string>{ path }, version)
{}

class SectorStreamReader
{
public:
  enum class H5Format : int8_t
  {
    Frame,
    DataCube
  };

  SectorStreamReader(const std::string& path);
  SectorStreamReader(const std::vector<std::string>& files);
  ~SectorStreamReader();

  Block read();

  // Reset to the start of the first file
  void reset();

  float dataCaptured();

  typedef BlockIterator<SectorStreamReader> iterator;
  iterator begin() { return iterator(this); }
  iterator end() { return iterator(nullptr); }
  void toHdf5(const std::string& path, H5Format format = H5Format::Frame);

private:
  struct Frame
  {
    Block block;
    int sectorCount = 0;
  };

  struct SectorStream
  {
    std::unique_ptr<std::ifstream> stream;
    int sector = -1;
    SectorStream(std::ifstream* str, int sec) : stream(str), sector(sec) {}
  };

  std::map<uint32_t, Frame> m_frameCache;
  std::vector<std::string> m_files;
  std::vector<SectorStream> m_streams;
  std::vector<SectorStream>::iterator m_streamsIterator;
  std::chrono::time_point<Clock> m_debugTimer;
  size_t m_totalFramesReconstructed = 0;
  size_t m_framesReconstructedAtLastPrint = 0;

  // Whether or not we are at the end of all of the files
  bool atEnd() const { return m_streams.empty(); }

  Header readHeader();
  Header readHeader(std::ifstream& stream);
  template <typename T>
  std::istream& read(T& value);
  template <typename T>
  std::istream& read(T* value, std::streamsize size);
  std::istream& skip(std::streamoff pos);
  template <typename T>
  std::istream& read(std::ifstream& stream, T& value);
  template <typename T>
  std::istream& read(std::ifstream& stream, T* value, std::streamsize size);

  void openFiles();
  template <typename Functor>
  void readAll(Functor f);
  void toHdf5FrameFormat(h5::H5ReadWrite& writer);
  void toHdf5DataCubeFormat(h5::H5ReadWrite& writer);
};

inline SectorStreamReader::SectorStreamReader(const std::string& path)
  : SectorStreamReader(std::vector<std::string>{ path })
{}
}

#endif
