#include "electron.h"
#include "reader.h"
#include "streamview.h"

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <emp/io/ContiguousStream.hpp>
#include <mpi.h>
#include <vector>

namespace stempy {

using SparseEventMap = std::map<uint32_t, std::vector<std::vector<uint32_t>>>;

void initMpiWorldRank(int& worldSize, int& rank)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
}

void serializeEvents(SparseEventMap& events, std::ostream* stream)
{
  cereal::BinaryOutputArchive archive(*stream);
  archive(events);
}

void serializeBlocks(std::vector<Block>& blocks, std::ostream* stream)
{
  cereal::BinaryOutputArchive archive(*stream);
  archive(blocks);
}

void receivePartialMap(Events& events, std::vector<char>& recvBuffer)
{
  MPI_Status status;
  // Probe for an incoming message from rank zero
  MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

  // Get the message size
  int msgSize;
  MPI_Get_count(&status, MPI_BYTE, &msgSize);

  // Resize our buffer
  recvBuffer.resize(msgSize);

  MPI_Recv(recvBuffer.data(), recvBuffer.size(), MPI_BYTE, MPI_ANY_SOURCE, 0,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  StreamView view(recvBuffer.data(), recvBuffer.size());
  std::istream stream(&view);
  cereal::BinaryInputArchive archive(stream);
  SparseEventMap remotePartialEventMap;
  archive(remotePartialEventMap);

  // Combine the events with our local events
  for (auto& remotePosEvents : remotePartialEventMap) {
    auto scanPosition = remotePosEvents.first;
    auto& remoteEvents = remotePosEvents.second;
    auto& localEvents = events[scanPosition];
    localEvents.insert(localEvents.end(),
                       std::make_move_iterator(remoteEvents.begin()),
                       std::make_move_iterator(remoteEvents.end()));
  }
}

void sendPartialMap(Events& events)
{
  // Create a partial map of electron events
  SparseEventMap partialEventMap;
  for (size_t i = 0; i < events.size(); i++) {
    auto& scanPosEvents = events[i];
    if (!scanPosEvents.empty()) {
      partialEventMap[i] = scanPosEvents;
    }
  }
  emp::ContiguousStream eventsContiguousStream;
  serializeEvents(partialEventMap, &eventsContiguousStream);

  if (eventsContiguousStream.GetSize() > INT_MAX) {
    throw std::runtime_error("Data too large to send using MPI.");
  }

  MPI_Send(eventsContiguousStream.GetData(), eventsContiguousStream.GetSize(),
           MPI_BYTE, 0, 0, MPI_COMM_WORLD);
}

void gatherEvents(int worldSize, int rank, Events& events)
{
  // Now the final gather!
  // It would be nice to use a gather here, but the displacement values can
  // overflow INT_MAX, so we do point-to-point. We will have to wait for MPI 4.0
  // and large count support.

  // On rank 0 receive and process the partial maps
  if (rank == 0) {
    // We pass this buffer in as we might be able to reduce the reallocations.
    std::vector<char> recvBuffer;
    // One receive for each other rank
    for (auto i = 0; i < worldSize - 1; i++) {
      receivePartialMap(events, recvBuffer);
    }
  }
  // Send partial maps to rank 0
  else {
    sendPartialMap(events);
  }
}

void gatherBlocks(int worldSize, int rank, std::vector<Block>& blocks)
{
  emp::ContiguousStream contiguousStream;
  serializeBlocks(blocks, &contiguousStream);

  int localSize = contiguousStream.GetSize();

  std::vector<int> sizesVec(worldSize);
  MPI_Gather(&localSize, 1, MPI_INT, sizesVec.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  std::vector<int> displacementsVec(worldSize);
  // Process them on rank 0
  std::vector<char> recvBuffer;
  char* recvData = nullptr;

  // On rank 0 calculate displacements and allocate memory for gather
  if (rank == 0) {
    // Work out the displacements
    int displacement = 0;
    for (size_t i = 0; i < displacementsVec.size(); i++) {
      displacementsVec[i] = displacement;
      displacement += sizesVec[i];
    }

    // Work out the total size
    size_t totalSize =
      std::accumulate(sizesVec.begin(), sizesVec.end(), static_cast<size_t>(0));

    // Check that we don't have an overflow
    if (totalSize > INT_MAX) {
      throw std::runtime_error("Data too large to send using MPI.");
    }

    // Allocate
    recvBuffer.resize(totalSize);
    recvData = recvBuffer.data();
  }

  // Now do the gather
  MPI_Gatherv(contiguousStream.GetData(), contiguousStream.GetSize(), MPI_BYTE,
              recvData, sizesVec.data(), displacementsVec.data(), MPI_BYTE, 0,
              MPI_COMM_WORLD);

  // On rank 0 deserialize the sample blocks
  if (rank == 0) {
    for (size_t i = 0; i < sizesVec.size(); i++) {
      // Skip over our own data we already have it!
      if (i == static_cast<size_t>(rank)) {
        continue;
      }

      StreamView view(recvData + displacementsVec[i], sizesVec[i]);
      std::istream stream(&view);
      cereal::BinaryInputArchive archive(stream);
      std::vector<Block> remoteSampleBlocks;
      archive(remoteSampleBlocks);
      blocks.insert(blocks.end(),
                    std::make_move_iterator(remoteSampleBlocks.begin()),
                    std::make_move_iterator(remoteSampleBlocks.end()));
    }
  }
}

void broadcastThresholds(double& background, double& xRay)
{
  std::vector<double> thresholds = { background, xRay };

  // Broadcast the thresholds
  MPI_Bcast(thresholds.data(), thresholds.size(), MPI_DOUBLE, 0,
            MPI_COMM_WORLD);

  background = thresholds[0];
  xRay = thresholds[1];
}

int getSampleBlocksPerRank(int worldSize, int rank, int thresholdNumberOfBlocks)
{
  auto sampleBlocksPerRank = thresholdNumberOfBlocks / worldSize;
  auto offset = rank * sampleBlocksPerRank;

  // We could be fairer but this is good for now
  if (rank == (worldSize - 1)) {
    sampleBlocksPerRank = thresholdNumberOfBlocks - offset;
  }

  return sampleBlocksPerRank;
}

} // namespace stempy
