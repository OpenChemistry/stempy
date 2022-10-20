#ifndef stempyreader_h
#define stempyreader_h

#include "config.h"

#include <ThreadPool.h>
#include <array>
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

#ifdef ENABLE_HDF5
namespace h5 {
class H5ReadWrite;
}
#endif // ENABLE_HDF5

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
  std::vector<bool> complete;

#ifdef USE_MPI
  template <class Archive>
  void serialize(Archive& archive);
#endif

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

#ifdef USE_MPI
  // Serialization methods for cereal
  template <class Archive>
  void save(Archive& ar) const;

  template <class Archive>
  void load(Archive& ar);
#endif

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
  short m_sector = -1;

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
  short sector() { return m_sector; };
};

inline StreamReader::StreamReader(const std::string& path, uint8_t version)
  : StreamReader(std::vector<std::string>{ path }, version)
{}

class SectorStreamReader
{
public:
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

#ifdef ENABLE_HDF5
  enum class H5Format : int8_t
  {
    Frame,
    DataCube
  };

  void toHdf5(const std::string& path, H5Format format = H5Format::Frame);
#endif // ENABLE_HDF5

  uint8_t version() const { return m_version; };

  struct SectorStream
  {
    std::unique_ptr<std::ifstream> stream;
    int sector = -1;
    // Mutex to guard access to the ifstream
    std::unique_ptr<std::mutex> mutex;
    SectorStream(std::ifstream* str, int sec)
      : stream(str), sector(sec), mutex(std::make_unique<std::mutex>()){};
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

#ifdef ENABLE_HDF5
  void toHdf5FrameFormat(h5::H5ReadWrite& writer);
  void toHdf5DataCubeFormat(h5::H5ReadWrite& writer);
#endif // ENABLE_HDF5

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

// struct to store a stream in the queue along with the appropriate metadata.
struct StreamQueueEntry
{
  StreamQueueEntry() = default;
  StreamQueueEntry(std::ifstream* str, int sec)
  {
    stream = str;
    sector = sec;
  }

  std::ifstream* stream = nullptr;
  int sector = -1;
  // Used for queue priority
  uint32_t readCount = 0;
};

// Compare type to encore order in the stream queue using the number of reads.
struct StreamQueueComparison
{
  bool reverse;

public:
  StreamQueueComparison(const bool& revparam = false) { reverse = revparam; }
  bool operator()(const StreamQueueEntry& lhs,
                  const StreamQueueEntry& rhs) const
  {
    if (reverse) {
      return (lhs.readCount < rhs.readCount);
    }
    else {
      return (lhs.readCount > rhs.readCount);
    }
  }
};

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

protected:
  // The number of threads to use
  int m_threads = -1;

  // The thread pool
  std::unique_ptr<ThreadPool> m_pool;

  // The futures associated with the worker threads
  std::vector<std::future<void>> m_futures;

private:
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

  // Queue of sector streams to be read by threads. We use a priority queue
  // based on the number of reads to ensure that we read all the files at the
  // same ratio. Using a round-robin approach didn't work on some platforms.
  std::mutex m_queueMutex;
  std::priority_queue<StreamQueueEntry, std::vector<StreamQueueEntry>,
                      StreamQueueComparison> m_streamQueue;

  void initNumberOfThreads();
  bool nextStream(StreamQueueEntry& entry);
};

template <typename Functor>
std::future<void> SectorStreamThreadedReader::readAll(Functor& func)
{
  m_pool = std::make_unique<ThreadPool>(m_threads);

  auto streamsIterator = m_streams.begin();

  while (streamsIterator != m_streams.end()) {
    auto& s = *streamsIterator;
    m_streamQueue.push(StreamQueueEntry(s.stream.get(), s.sector));
    streamsIterator++;
  }

  // Create worker threads
  for (int i = 0; i < m_threads; i++) {
    m_futures.emplace_back(m_pool->enqueue([this, &func]() {

      while (!m_streams.empty()) {
        // Get the next stream to read from
        StreamQueueEntry streamQueueEntry;

        if (!nextStream(streamQueueEntry)) {
          continue;
        }
        auto& stream = streamQueueEntry.stream;
        auto sector = streamQueueEntry.sector;

        // First read the header
        auto header = readHeader(*stream);

        std::vector<Block> blocks;
        for (unsigned j = 0; j < header.imagesInBlock; j++) {
          auto pos = header.imageNumbers[j];
          auto frameNumber = header.frameNumber;

          std::unique_lock<std::mutex> cacheLock(m_cacheMutex);
          auto& frame = m_frameCache[frameNumber];
          cacheLock.unlock();

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
            cacheLock.lock();
            blocks.emplace_back(frame.block);
            m_frameCache.erase(frameNumber);
            cacheLock.unlock();
          }
        }

        // Return the stream to the queue so other threads can read from it.
        // It is important that we do this before doing the processing to prevent
        // starvation of one of the streams, we need to make sure they are all
        // read evenly.
        {
          std::unique_lock<std::mutex> queueLock(m_queueMutex);
          streamQueueEntry.readCount++;
          m_streamQueue.push(streamQueueEntry);
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

// struct to hold the location of a sector: sector, stream and offset
struct SectorLocation
{
  short sector = -1;
#ifdef USE_MPI
  int streamIndex = -1;
  // Serialization method for cereal
  template <class Archive>
  void serialize(Archive& archive);
#endif
  SectorStreamReader::SectorStream* sectorStream = nullptr;
  std::streampos offset;
};

// This type holds the locations of the sectors for each frame at a give scan
// position. Note: That there can be multiple frames at a give position, we key
// the map of the frame number.
using ScanMap = std::vector<std::map<uint32_t, std::array<SectorLocation, 4>>>;

// SectorStreamMultiPassThreadedReader uses a two pass approach to frame
// reconstruction. First is reads the header from all sectors and uses them
// to build up "frame maps" detailing the streams and offsets for all the
// sectors in a frame. The second pass is used reconstruct the frames using this
// map. This avoids the need to cache partial frames so used less memory.
// Depending on the seek performance of the disks it also performs well.
class SectorStreamMultiPassThreadedReader : public SectorStreamThreadedReader
{
public:
  SectorStreamMultiPassThreadedReader(const std::string& path,
                                      int threads = -1);
  SectorStreamMultiPassThreadedReader(const std::vector<std::string>& files,
                                      int threads = -1);

  template <typename Functor>
  std::future<void> readAll(Functor& f);

private:
  ScanMap m_scanMap;
  uint32_t m_scanMapOffset = 0;
  uint32_t m_scanMapSize = 0;
  uint32_t m_streamsOffset = 0;
  uint32_t m_streamsSize = 0;
  // atomic to keep track of the header or frame being processed
  std::atomic<uint32_t> m_processed = { 0 };


  // Mutex to lock the map of frames at each scan position
  std::vector<std::unique_ptr<std::mutex>> m_scanPositionMutexes;

  void readHeaders();
  template <typename Functor>
  void processFrames(Functor& func, Header header);

#ifdef USE_MPI
  int m_rank;
  int m_worldSize;

  void initMPI();
  void serializeScanMap(std::ostream* stream);
  void partitionScanMap();
  void gatherScanMap();

#endif
};

// Read the FrameMaps for scan and reconstruct the frame before performing the
// processing
template <typename Functor>
void SectorStreamMultiPassThreadedReader::processFrames(Functor& func,
                                                        Header header)
{
  while (m_processed < m_scanMapOffset + m_scanMapSize) {
    auto imageNumber = m_processed++;

    // Need to check we haven't over run
    if (imageNumber >= m_scanMapOffset + m_scanMapSize) {
      break;
    }

    auto& frameMaps = m_scanMap[imageNumber];

    // Iterate over frame maps for this scan position
    for (const auto& f : frameMaps) {
      auto frameNumber = f.first;
      auto& frameMap = f.second;

      Block b;
      b.header.version = version();
      b.header.scanNumber = header.scanNumber;
      b.header.scanDimensions = header.scanDimensions;
      b.header.imagesInBlock = 1;
      b.header.frameNumber = frameNumber;
      b.header.imageNumbers.resize(1);
      b.header.imageNumbers[0] = imageNumber;
      b.header.complete.resize(1);

      b.header.frameDimensions = FRAME_DIMENSIONS;

      b.data.reset(new uint16_t[b.header.frameDimensions.first *
                                b.header.frameDimensions.second],
                   std::default_delete<uint16_t[]>());
      std::fill(b.data.get(),
                b.data.get() + b.header.frameDimensions.first *
                                 b.header.frameDimensions.second,
                0);

      short sectors = 0;
      for (int j = 0; j < 4; j++) {
        auto& sectorLocation = frameMap[j];

        if (sectorLocation.sectorStream != nullptr) {
          auto sectorStream = sectorLocation.sectorStream;
          std::unique_lock<std::mutex> lock(*sectorStream->mutex.get());
          sectorStream->stream->seekg(sectorLocation.offset);
          readSectorData(*sectorStream->stream, b, j);
          sectors++;
        }
      }

      // Mark if the frame is complete
      b.header.complete[0] = sectors == 4;

      // Finally process the frame
      func(b);
    }
  }
}

template <typename Functor>
std::future<void> SectorStreamMultiPassThreadedReader::readAll(Functor& func)
{
  m_pool = std::make_unique<ThreadPool>(m_threads);

  // Read one header to get scan size
  auto stream = m_streams[0].stream.get();
  auto header = readHeader(*stream);
  // Reset the stream
  stream->seekg(0);

  // Resize the vector to hold the frame sector locations for the scan
  m_scanMapSize = header.scanDimensions.first * header.scanDimensions.second;
  m_scanMap.clear();
  m_scanMap.resize(m_scanMapSize);

  // Allocate the mutexes
  m_scanPositionMutexes.clear();
  for (unsigned i = 0; i < m_scanMapSize; i++) {
    m_scanPositionMutexes.push_back(std::make_unique<std::mutex>());
  }

  // Reset counter
  m_processed = m_streamsOffset;

  // Enqueue lambda's to read headers to build up the locations of the sectors
  for (int i = 0; i < m_threads; i++) {
    m_futures.emplace_back(m_pool->enqueue([this]() { readHeaders(); }));
  }

  // Wait for all files to be processed
  for (auto& future : this->m_futures) {
    future.get();
  }

#ifdef USE_MPI
  gatherScanMap();

  // Partion scan map
  partitionScanMap();
#endif

  // Reset the streams
  for (auto& sectorStream : this->m_streams) {
    sectorStream.stream->seekg(0);
  }

  m_futures.clear();

  // Reset counter
  m_processed = m_scanMapOffset;

  // Now enqueue lambda's to read the frames and run processing
  for (int i = 0; i < m_threads; i++) {
    m_futures.emplace_back(m_pool->enqueue(
      [this, &func, header]() { processFrames(func, header); }));
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
