#ifndef stempyreader_h
#define stempyreader_h

#include <ThreadPool.h>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

namespace h5 {
class H5ReadWrite;
}

namespace stempy {

// Convention is (x, y)
using Coordinates2D = std::pair<int, int>;

// Convention is (width, height)
using Dimensions2D = std::pair<uint32_t, uint32_t>;

/// Currently the detector data is written in four sectors
const Dimensions2D SECTOR_DIMENSIONS_VERSION_4 = { 144, 576 };
const Dimensions2D SECTOR_DIMENSIONS_VERSION_5 = { 576, 144 };

/// This is the detector size at  NCEM
const Dimensions2D FRAME_DIMENSIONS = { 576, 576 };

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

  SectorStreamReader(const std::string& path, uint8_t version = 5);
  SectorStreamReader(const std::vector<std::string>& files,
                     uint8_t version = 5);
  ~SectorStreamReader();

  Block read();
  template <typename Functor>
  void readAll(Functor f);

  // Reset to the start of the first file
  void reset();

  float dataCaptured();

  typedef BlockIterator<SectorStreamReader> iterator;
  iterator begin() { return iterator(this); }
  iterator end() { return iterator(nullptr); }
  void toHdf5(const std::string& path, H5Format format = H5Format::Frame);
  uint8_t version() const { return m_version; };

  struct SectorStream
  {
    std::unique_ptr<std::ifstream> stream;
    int sector = -1;
    SectorStream(std::ifstream* str, int sec) : stream(str), sector(sec){};
  };

protected:
  struct Frame
  {
    Block block;
    std::atomic<int> sectorCount = { 0 };
    std::mutex mutex;
  };
  std::map<uint32_t, Frame> m_frameCache;
  std::vector<SectorStream> m_streams;

  Header readHeader(std::ifstream& stream);
  void readSectorData(std::ifstream& stream, Block& block, int sector);

private:
  std::vector<std::string> m_files;

  std::vector<SectorStream>::iterator m_streamsIterator;

  uint8_t m_version;

  // Whether or not we are at the end of all of the files
  bool atEnd() const { return m_streams.empty(); }

  Header readHeader();
  void readSectorData(Block& block, int sector);
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
  void toHdf5FrameFormat(h5::H5ReadWrite& writer);
  void toHdf5DataCubeFormat(h5::H5ReadWrite& writer);
  void readSectorDataVersion4(std::ifstream& stream, Block& block, int sector);
  void readSectorDataVersion5(std::ifstream& stream, Block& block, int sector);
};

inline SectorStreamReader::SectorStreamReader(const std::string& path,
                                              uint8_t version)
  : SectorStreamReader(std::vector<std::string>{ path }, version)
{}

class ElectronCountedData;
template <typename T>
class Image;

using SectorStreamPair = std::pair<std::ifstream*, int>;

class SectorStreamThreadedReader : public SectorStreamReader
{
public:
  SectorStreamThreadedReader(const std::string& path, uint8_t version = 5);
  SectorStreamThreadedReader(const std::vector<std::string>& files,
                             uint8_t version = 5);
  SectorStreamThreadedReader(const std::string& path, uint8_t version = 5,
                             int threads = 0);
  SectorStreamThreadedReader(const std::vector<std::string>& files,
                             uint8_t version = 5, int threads = 0);

  template <typename Functor>
  std::future<void> readAll(Functor& f);

private:
  // The number of threads to use
  int m_threads = -1;

  // The thread pool
  std::unique_ptr<ThreadPool> m_pool;

  // The futures associated with the worker threads
  std::vector<std::future<void>> m_futures;

  // Protect access to frame cache
  std::mutex m_cacheMutex;

  // Protect access to the streams
  std::mutex m_streamsMutex;

  // Protect access to samples
  std::mutex m_sampleMutex;

  // Control collection of samples
  std::condition_variable m_sampleCondition;

  // The samples to use for calculating the thresholds
  std::vector<Block> m_sampleBlocks;

  // Queue of sector streams to be read by threads
  std::mutex m_queueMutex;
  std::queue<SectorStreamPair> m_streamQueue;

  void initNumberOfThreads();
  bool nextSectorStreamPair(SectorStreamPair& pair);
};

template <typename Functor>
std::future<void> SectorStreamThreadedReader::readAll(Functor& func)
{
  m_pool = std::make_unique<ThreadPool>(m_threads);

  auto streamsIterator = m_streams.begin();

  while (streamsIterator != m_streams.end()) {
    auto& s = *streamsIterator;
    m_streamQueue.push(std::make_pair(s.stream.get(), s.sector));
    streamsIterator++;
  }

  // Create worker threads
  for (int i = 0; i < m_threads; i++) {
    m_futures.emplace_back(m_pool->enqueue([this, &func]() {

      while (!m_streams.empty()) {
        // Get the next stream to read from
        SectorStreamPair sectorStreamPair;
        if (!nextSectorStreamPair(sectorStreamPair)) {
          continue;
        }
        auto& stream = sectorStreamPair.first;
        auto sector = sectorStreamPair.second;

        // First read the header
        auto header = readHeader(*stream);

        std::vector<Block> blocks;
        for (unsigned j = 0; j < header.imagesInBlock; j++) {
          auto pos = header.imageNumbers[j];
          auto frameNumber = header.frameNumber;

          std::unique_lock<std::mutex> mutexLock(m_cacheMutex);
          auto& frame = m_frameCache[frameNumber];
          mutexLock.unlock();

          // Do we need to allocate the frame, use a double check lock
          if (std::atomic_load(&frame.block.data) == nullptr) {
            std::unique_lock<std::mutex> lock(frame.mutex);
            // Check again now we have the mutex
            if (std::atomic_load(&frame.block.data) == nullptr) {
              frame.block.header.version = version();
              frame.block.header.scanNumber = header.scanNumber;
              frame.block.header.scanDimensions = header.scanDimensions;
              frame.block.header.imagesInBlock = 1;
              frame.block.header.imageNumbers.push_back(pos);
              frame.block.header.frameNumber = frameNumber;
              frame.block.header.frameDimensions = FRAME_DIMENSIONS;
              std::shared_ptr<uint16_t> data;

              data.reset(
                new uint16_t[frame.block.header.frameDimensions.first *
                             frame.block.header.frameDimensions.second],
                std::default_delete<uint16_t[]>());
              std::fill(data.get(),
                        data.get() +
                          frame.block.header.frameDimensions.first *
                            frame.block.header.frameDimensions.second,
                        0);
              std::atomic_store(&frame.block.data, data);
            }
          }

          readSectorData(*stream, frame.block, sector);

          // Now now have the complete frame
          if (++frame.sectorCount == 4) {
            mutexLock.lock();
            blocks.emplace_back(frame.block);
            m_frameCache.erase(frameNumber);
            mutexLock.unlock();
          }
        }

        // Return the stream to the queue so other threads can read from it.
        // It is important that we do this before doing the processing to prevent
        // starvation of one of the streams, we need to make sure they are all
        // read evenly.
        {
          std::unique_lock<std::mutex> queueLock(m_queueMutex);
          m_streamQueue.push(sectorStreamPair);
        }

        // Finally call the function on any completed frames
        for (auto& b : blocks) {
          func(b);
        }
      }
    }));
  }

  // Return a future that is resolved once the processing is complete
  auto complete = std::async(std::launch::deferred, [this]() {
    for (auto& future : this->m_futures) {
      future.get();
    }
  });

  return complete;
}
}

#endif
