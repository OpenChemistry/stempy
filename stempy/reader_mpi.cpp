#include "reader.h"
#include "streamview.h"

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <emp/io/ContiguousStream.hpp>
#include <mpi.h>
#include <numeric>
#include <vector>

namespace emp {
class ContiguousStream;
}

template <class Archive>
void save(Archive& archive, std::streampos const& m)
{
  // Not sure of the best what to serialze a std::streampos, this seems to
  // work, but may not be portable or work with all file sizes.
  int64_t pos = static_cast<int64_t>(m);
  archive.saveBinary(&pos, sizeof(int64_t));
}

template <class Archive>
void load(Archive& archive, std::streampos& m)
{
  int64_t p;
  archive.loadBinary(&p, sizeof(int64_t));
  m = static_cast<std::streampos>(p);
}

// Needed to support std::pairs
// https://github.com/USCiLab/cereal/issues/547
namespace cereal {
template <class Archive, class F, class S>
void save(Archive& ar, const std::pair<F, S>& pair)
{
  ar(pair.first, pair.second);
}

template <class Archive, class F, class S>
void load(Archive& ar, std::pair<F, S>& pair)
{
  ar(pair.first, pair.second);
}

template <class Archive, class F, class S>
struct specialize<Archive, std::pair<F, S>,
                  cereal::specialization::non_member_load_save>
{};
} // namespace cereal

namespace stempy {

// Sparse scan map
using SparseScanMap =
  std::map<uint32_t, std::map<uint32_t, std::array<SectorLocation, 4>>>;

template <class Archive>
void Header::serialize(Archive& archive)
{
  archive(scanDimensions, frameDimensions, imagesInBlock, frameNumber,
          imageNumbers);
}

template <class Archive>
void Block::save(Archive& ar) const
{
  ar(header);
  ar.saveBinary(data.get(), sizeof(uint16_t) * header.frameDimensions.first *
                              header.frameDimensions.second *
                              header.imagesInBlock);
}

template <class Archive>
void Block::load(Archive& ar)
{
  ar(header);
  auto size = header.frameDimensions.first * header.frameDimensions.second *
              header.imagesInBlock;
  data.reset(new uint16_t[size], std::default_delete<uint16_t[]>());
  ar.loadBinary(data.get(), sizeof(uint16_t) * size);
}

template <class Archive>
void SectorLocation::serialize(Archive& archive)
{
  archive(sector, streamIndex, offset);
}

void SectorStreamMultiPassThreadedReader::initMPI()
{
  int init = 0;
  MPI_Initialized(&init);
  if (!init) {
    MPI_Init(NULL, NULL);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &m_worldSize);
  auto numberOfFiles = m_streams.size();

  auto filesPerRank = numberOfFiles / m_worldSize;

  m_streamsOffset = m_rank * filesPerRank;
  m_streamsSize = filesPerRank;

  // For now the last rank will get some extra, we can allocate more equally in
  // the future.
  if (m_rank == (m_worldSize - 1)) {
    m_streamsSize = numberOfFiles - m_streamsOffset;
  }
}

void SectorStreamMultiPassThreadedReader::serializeScanMap(std::ostream* stream)
{
  SparseScanMap partialMap;
  for (size_t i = 0; i < m_scanMap.size(); i++) {
    if (!m_scanMap[i].empty()) {
      partialMap[i] = m_scanMap[i];
    }
  }

  cereal::BinaryOutputArchive archive(*stream);
  archive(partialMap);
}

void SectorStreamMultiPassThreadedReader::partitionScanMap()
{
  auto scanPositions = m_scanMap.size();

  auto scanPositionsPerRank = scanPositions / m_worldSize;
  m_scanMapOffset = m_rank * scanPositionsPerRank;
  m_scanMapSize = scanPositionsPerRank;

  // We could be fairer but this is good for now
  if (m_rank == (m_worldSize - 1)) {
    m_scanMapSize = scanPositions - m_scanMapOffset;
  }
}

void SectorStreamMultiPassThreadedReader::gatherScanMap()
{

  emp::ContiguousStream contiguousStream;
  serializeScanMap(&contiguousStream);

  int localSize = contiguousStream.GetSize();

  // First send the buffer sizes
  std::vector<int> sizes(m_worldSize);
  MPI_Allgather(&localSize, 1, MPI_INT, sizes.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  // Work out the displacements
  std::vector<int> displacements(m_worldSize);
  int displacement = 0;
  for (size_t i = 0; i < displacements.size(); i++) {
    displacements[i] = displacement;
    displacement += sizes[i];
  }

  // Work out the total size
  size_t totalSize =
    std::accumulate(sizes.begin(), sizes.end(), static_cast<size_t>(0));

  // Allocate
  std::vector<char> recvData(totalSize);

  // Now do the gather
  MPI_Allgatherv(contiguousStream.GetData(), contiguousStream.GetSize(),
                 MPI_BYTE, recvData.data(), sizes.data(), displacements.data(),
                 MPI_BYTE, MPI_COMM_WORLD);

  // Now deserialize our partial scan maps
  for (size_t i = 0; i < sizes.size(); i++) {
    // Skip over our own data we already have it!
    if (i == static_cast<size_t>(m_rank)) {
      continue;
    }

    StreamView view(recvData.data() + displacements[i], sizes[i]);
    std::istream stream(&view);
    cereal::BinaryInputArchive archive(stream);
    SparseScanMap partialMap;
    archive(partialMap);

    // Process the partial map and update our local copy
    for (const auto& m : partialMap) {
      auto scanPosition = m.first;

      auto frameMaps = m.second;
      for (const auto& f : frameMaps) {
        auto frameNumber = f.first;
        auto frameMap = f.second;
        for (int j = 0; j < 4; j++) {

          auto sectorLocation = frameMap[j];
          if (sectorLocation.streamIndex != -1) {

            // Initialize the sector stream base on the index
            auto& sectorStream = m_streams[sectorLocation.streamIndex];
            sectorLocation.sectorStream = &sectorStream;
            auto& frames = m_scanMap[scanPosition];
            auto& location = frames[frameNumber];
            location[j] = sectorLocation;
          }
        }
      }
    }
  }
}

template void Block::load<cereal::BinaryInputArchive>(
  cereal::BinaryInputArchive&);
template void Block::save<cereal::BinaryOutputArchive>(
  cereal::BinaryOutputArchive&) const;

} // namespace stempy
