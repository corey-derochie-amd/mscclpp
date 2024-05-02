// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <unordered_map>
#include <vector>

#include "allgather.hpp"
#include "allreduce.hpp"
#include "mscclpp_nccl.h"

#define MSCCLPP_NCCL_API extern "C" __attribute__((visibility("default")))

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

#define NUM_CHANNELS_PER_CONNECTION 64
__device__ mscclpp::DeviceSyncer deviceSyncer;

// static const mscclpp::Transport IBs[] = {mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2,
//                             mscclpp::Transport::IB3, mscclpp::Transport::IB4, mscclpp::Transport::IB5,
//                             mscclpp::Transport::IB6, mscclpp::Transport::IB7};

struct channelKey {
  const void* sendbuff;
  const void* recvbuff;
  size_t bytes;
  bool operator==(const channelKey& other) const {
    return sendbuff == other.sendbuff && recvbuff == other.recvbuff && bytes == other.bytes;
  }
};

namespace std {
template <>
struct hash<channelKey> {
  std::size_t operator()(const channelKey& k) const {
    return std::hash<const void*>()(k.sendbuff) ^ std::hash<const void*>()(k.recvbuff) ^ std::hash<size_t>()(k.bytes);
  }
};
}  // namespace std

struct ChannelInfo {
  std::vector<mscclpp::SmChannel> smChannels;
  std::vector<mscclpp::SmChannel> smOutChannels;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> smOutChannelDeviceHandles;
};

struct mscclpp_ncclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;

  std::unordered_map<channelKey, ChannelInfo> channelInfos;
  std::shared_ptr<char> scratchBuff;
  std::vector<mscclpp::RegisteredMemory> remoteScratchRegMemories;
};

static size_t mscclpp_ncclTypeSize(mscclpp_ncclDataType_t type) {
  switch (type) {
    case mscclpp_ncclInt8:
    case mscclpp_ncclUint8:
      return 1;
    case mscclpp_ncclFloat16:
      return 2;
    case mscclpp_ncclInt32:
    case mscclpp_ncclUint32:
      return 4;
    case mscclpp_ncclInt64:
    case mscclpp_ncclUint64:
      return 8;
    case mscclpp_ncclFloat32:
      return 4;
    case mscclpp_ncclFloat64:
      return 8;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case mscclpp_ncclBfloat16:
      return 2;
#endif  // defined(__CUDA_BF16_TYPES_EXIST__)
#if defined(__CUDA_FP8_TYPES_EXIST__)
    case mscclpp_ncclFp8E4M3:
    case mscclpp_ncclFp8E5M2:
      return 1;
#endif  // defined(__CUDA_FP8_TYPES_EXIST__)
    case mscclpp_ncclNumTypes:
      return 0;
  }
  return 0;
}

static mscclpp::Transport getTransport(int, int) {
  // if (rank / nRanksPerNode == peerRank / nRanksPerNode) {
  //   return mscclpp::Transport::CudaIpc;
  // } else {
  //   return IBs[rank % nRanksPerNode];
  // }
  return mscclpp::Transport::CudaIpc;
}

static std::vector<mscclpp::RegisteredMemory> setupRemoteMemories(std::shared_ptr<mscclpp::Communicator> comm, int rank,
                                                                  void* buff, size_t bytes,
                                                                  mscclpp::TransportFlags transport) {
  std::vector<mscclpp::RegisteredMemory> remoteMemories;
  mscclpp::RegisteredMemory memory = comm->registerMemory(buff, bytes, transport);
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemoryFutures;
  for (int i = 0; i < comm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    remoteRegMemoryFutures.push_back(comm->recvMemoryOnSetup(i, 0));
    comm->sendMemoryOnSetup(memory, i, 0);
  }
  comm->setup();
  std::transform(remoteRegMemoryFutures.begin(), remoteRegMemoryFutures.end(), std::back_inserter(remoteMemories),
                 [](const auto& future) { return future.get(); });
  return remoteMemories;
}

static std::vector<mscclpp::SmChannel> setupSmChannels(mscclpp_ncclComm_t comm,
                                                       const std::vector<mscclpp::RegisteredMemory>& remoteMemories,
                                                       void* src) {
  std::vector<mscclpp::SmChannel> channels;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>& smSemaphores = comm->smSemaphores;
  size_t nConnections = comm->connections.size();
  for (size_t idx = 0; idx < NUM_CHANNELS_PER_CONNECTION; ++idx) {
    for (size_t cid = 0; cid < nConnections; ++cid) {
      if (comm->connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(smSemaphores[idx * nConnections + cid], remoteMemories[cid], src, nullptr);
      }
    }
  }
  return channels;
}

static std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> setupSmChannelDeviceHandles(
    const std::vector<mscclpp::SmChannel>& smChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::transform(smChannels.begin(), smChannels.end(), std::back_inserter(smChannelDeviceHandles),
                 [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> ptr =
      mscclpp::allocSharedCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(smChannelDeviceHandles.size());
  mscclpp::AvoidCudaGraphCaptureGuard guard;
  CUDACHECK(cudaMemcpy(ptr.get(), smChannelDeviceHandles.data(),
                       sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>) * smChannelDeviceHandles.size(),
                       cudaMemcpyHostToDevice));
  return ptr;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclGetVersion(int* version) {
  if (version == nullptr) return mscclpp_ncclInvalidArgument;
  *version = MSCCLPP_VERSION;
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclGetUniqueId(mscclpp_ncclUniqueId* uniqueId) {
  if (uniqueId == nullptr) return mscclpp_ncclInvalidArgument;
  if (MSCCLPP_UNIQUE_ID_BYTES != MSCCLPP_NCCL_UNIQUE_ID_BYTES) return mscclpp_ncclInternalError;
  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();
  memcpy(uniqueId, &id, sizeof(mscclpp_ncclUniqueId));
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommInitRankConfig(mscclpp_ncclComm_t* comm, int nranks, mscclpp_ncclUniqueId commId, int rank,
                                             mscclpp_ncclConfig_t* config) {
  // TODO: implement this function
  //return mscclpp_ncclInternalError;
  return mscclpp_ncclCommInitRank(comm, nranks, commId, rank);
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommInitRank(mscclpp_ncclComm_t* comm, int nranks, mscclpp_ncclUniqueId commId, int rank) {
  if (comm == nullptr) return mscclpp_ncclInvalidArgument;
  if (nranks < 0 || rank < 0 || rank >= nranks) return mscclpp_ncclInvalidArgument;
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
  mscclpp::UniqueId id;
  memcpy(id.data(), &commId, sizeof(mscclpp_ncclUniqueId));
  bootstrap->initialize(id);
  std::shared_ptr<mscclpp::Communicator> mscclppComm = std::make_shared<mscclpp::Communicator>(bootstrap);
  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;

  for (int i = 0; i < mscclppComm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    mscclpp::Transport transport = getTransport(rank, i);
    connectionFutures.push_back(mscclppComm->connectOnSetup(i, 0, transport));
  }
  mscclppComm->setup();

  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                 [](const auto& future) { return future.get(); });

  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
  for (size_t idx = 0; idx < NUM_CHANNELS_PER_CONNECTION; ++idx) {
    for (size_t cid = 0; cid < connections.size(); ++cid) {
      if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        smSemaphores.emplace_back(
            std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*(mscclppComm), connections[cid]));
      }
    }
  }
  mscclppComm->setup();

  mscclpp_ncclComm* commPtr = new mscclpp_ncclComm();
  commPtr->comm = mscclppComm;
  commPtr->connections = std::move(connections);
  commPtr->smSemaphores = std::move(smSemaphores);
  commPtr->scratchBuff = mscclpp::allocExtSharedCuda<char>(SCRATCH_SIZE);
  commPtr->remoteScratchRegMemories =
      setupRemoteMemories(commPtr->comm, rank, commPtr->scratchBuff.get(), SCRATCH_SIZE, mscclpp::Transport::CudaIpc);

  *comm = commPtr;
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommInitAll(mscclpp_ncclComm_t*, int, const int*) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommFinalize(mscclpp_ncclComm_t comm) {
  comm->comm->bootstrap()->barrier();
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommDestroy(mscclpp_ncclComm_t comm) {
  if (comm == nullptr) return mscclpp_ncclInvalidArgument;
  delete comm;
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommAbort(mscclpp_ncclComm_t) {
  // TODO: implement this function
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommSplit(mscclpp_ncclComm_t, int, int, mscclpp_ncclComm_t*, mscclpp_ncclConfig_t*) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API const char* mscclpp_ncclGetErrorString(mscclpp_ncclResult_t result) {
  switch (result) {
    case mscclpp_ncclSuccess:
      return "no error";
    case mscclpp_ncclUnhandledCudaError:
      return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case mscclpp_ncclSystemError:
      return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case mscclpp_ncclInternalError:
      return "internal error - please report this issue to the NCCL developers";
    case mscclpp_ncclInvalidArgument:
      return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case mscclpp_ncclInvalidUsage:
      return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case mscclpp_ncclRemoteError:
      return "remote process exited or there was a network error";
    case mscclpp_ncclInProgress:
      return "NCCL operation in progress";
    default:
      return "unknown result code";
  }
}

MSCCLPP_NCCL_API const char* mscclpp_ncclGetLastError(mscclpp_ncclComm_t) {
  // TODO: implement this function
  
  //return nullptr;
  char *s = "success";
  return s;

}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommGetAsyncError(mscclpp_ncclComm_t, mscclpp_ncclResult_t* asyncError) {
  if (asyncError == nullptr) return mscclpp_ncclInvalidArgument;
  *asyncError = mscclpp_ncclSuccess;
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommCount(const mscclpp_ncclComm_t comm, int* count) {
  if (comm == nullptr || count == nullptr) return mscclpp_ncclInvalidArgument;
  *count = comm->comm->bootstrap()->getNranks();
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommCuDevice(const mscclpp_ncclComm_t comm, int* device) {
  if (comm == nullptr || device == nullptr) return mscclpp_ncclInvalidArgument;
  *device = comm->comm->bootstrap()->getRank();
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclCommUserRank(const mscclpp_ncclComm_t comm, int* rank) {
  if (comm == nullptr || rank == nullptr) return mscclpp_ncclInvalidArgument;
  *rank = comm->comm->bootstrap()->getRank();
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclRedOpCreatePreMulSum(mscclpp_ncclRedOp_t*, void*, mscclpp_ncclDataType_t,
                                               mscclpp_ncclScalarResidence_t, mscclpp_ncclComm_t) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclRedOpDestroy(mscclpp_ncclRedOp_t, mscclpp_ncclComm_t) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclReduce(const void*, void*, size_t, mscclpp_ncclDataType_t,
                                 mscclpp_ncclRedOp_t, int, mscclpp_ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclBcast(void*, size_t, mscclpp_ncclDataType_t, int, mscclpp_ncclComm_t,
                                cudaStream_t) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclBroadcast(const void*, void*, size_t, mscclpp_ncclDataType_t,
                                    int, mscclpp_ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, mscclpp_ncclDataType_t datatype,
                                    mscclpp_ncclRedOp_t, mscclpp_ncclComm_t comm, cudaStream_t stream) {
  if (count < 8)
          count =  8;
  size_t bytes = count * mscclpp_ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) return mscclpp_ncclInvalidArgument;
  int rank = comm->comm->bootstrap()->getRank();
  channelKey key{sendbuff, recvbuff, bytes};
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels = nullptr;

  auto it = comm->channelInfos.find(key);
  if (it == comm->channelInfos.end()) {
    // setup smChannels (src: sendbuff, dst: remote scratch buff)
    std::vector<mscclpp::SmChannel> channels = setupSmChannels(comm, comm->remoteScratchRegMemories, const_cast<void*>(sendbuff));
    ChannelInfo channelInfo{channels, {}, setupSmChannelDeviceHandles(channels), nullptr};
    it = comm->channelInfos.emplace(key, channelInfo).first;

    // setup smOutChannels (src: recvbuff, dst: remote recvbuff)
    if (bytes > (1 << 20)) {
    //// Increasing to 64MB from 1MB
    // if (bytes > (1 << 26)) {
      std::vector<mscclpp::RegisteredMemory> remoteMemories =
          setupRemoteMemories(comm->comm, rank, recvbuff, bytes, mscclpp::Transport::CudaIpc);
      std::vector<mscclpp::SmChannel> outChannels = setupSmChannels(comm, remoteMemories, recvbuff);
      it->second.smOutChannels = outChannels;
      it->second.smOutChannelDeviceHandles = setupSmChannelDeviceHandles(outChannels);
    }
  }

  smChannels = it->second.smChannelDeviceHandles.get();
  smOutChannels = it->second.smOutChannelDeviceHandles.get();

  switch (datatype) {
    case mscclpp_ncclFloat16:
      CUDACHECK(allreduce((half*)sendbuff, (half*)comm->scratchBuff.get(), (half*)recvbuff, smChannels, smOutChannels,
                          rank, NRANKS_PER_NODE, comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    case mscclpp_ncclFloat32:
      CUDACHECK(allreduce((float*)sendbuff, (float*)comm->scratchBuff.get(), (float*)recvbuff, smChannels,
                          smOutChannels, comm->comm->bootstrap()->getRank(), NRANKS_PER_NODE,
                          comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    case mscclpp_ncclInt32:
    case mscclpp_ncclUint32:
      CUDACHECK(allreduce((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels, smOutChannels,
                          comm->comm->bootstrap()->getRank(), NRANKS_PER_NODE, comm->comm->bootstrap()->getNranks(),
                          count, stream));
      break;
    default:
      return mscclpp_ncclInvalidArgument;
  }
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclReduceScatter(const void*, void*, size_t, mscclpp_ncclDataType_t,
                                        mscclpp_ncclRedOp_t, mscclpp_ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, mscclpp_ncclDataType_t datatype,
                                    mscclpp_ncclComm_t comm, cudaStream_t stream) {
  size_t bytes = sendcount * mscclpp_ncclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr) return mscclpp_ncclInvalidArgument;
  int rank = comm->comm->bootstrap()->getRank();
  int nRank = comm->comm->bootstrap()->getNranks();
  channelKey key{sendbuff, recvbuff, bytes};
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;

  auto it = comm->channelInfos.find(key);
  if (it == comm->channelInfos.end()) {
    std::vector<mscclpp::RegisteredMemory> remoteMemories =
        setupRemoteMemories(comm->comm, rank, const_cast<void*>(recvbuff), bytes * nRank,
                            mscclpp::Transport::CudaIpc);
    std::vector<mscclpp::SmChannel> channels =
        setupSmChannels(comm, remoteMemories, const_cast<void*>(recvbuff));
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
    std::transform(channels.begin(), channels.end(), std::back_inserter(smChannelDeviceHandles),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    ChannelInfo channelInfo{channels, {}, setupSmChannelDeviceHandles(channels), nullptr};
    it = comm->channelInfos.emplace(key, channelInfo).first;
  }
  smChannels = it->second.smChannelDeviceHandles.get();
  if ((char*)sendbuff == (char*)recvbuff + rank * sendcount) {
    CUDACHECK(allgather<false>((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels,
                        rank, NRANKS_PER_NODE, nRank, bytes / sizeof(int), stream));
  } else {
    CUDACHECK(allgather<true>((int*)sendbuff, (int*)comm->scratchBuff.get(), (int*)recvbuff, smChannels,
                        rank, NRANKS_PER_NODE, nRank, bytes / sizeof(int), stream));
  }
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclSend(const void*, size_t, mscclpp_ncclDataType_t, int, mscclpp_ncclComm_t,
                               cudaStream_t) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclRecv(void*, size_t, mscclpp_ncclDataType_t, int, mscclpp_ncclComm_t,
                               cudaStream_t) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclAllToAll(const void*, void*, size_t, mscclpp_ncclDataType_t,
                                   mscclpp_ncclComm_t, cudaStream_t) {
  // TODO: implement this function
  return mscclpp_ncclInternalError;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclGroupStart() {
  // Do nothing
  return mscclpp_ncclSuccess;
}

MSCCLPP_NCCL_API mscclpp_ncclResult_t mscclpp_ncclGroupEnd() {
  // Do nothing
  return mscclpp_ncclSuccess;
}
