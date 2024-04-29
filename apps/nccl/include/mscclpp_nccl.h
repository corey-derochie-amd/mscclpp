/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_NCCL_H_
#define MSCCLPP_NCCL_H_

#include <mscclpp/gpu.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>
/* Opaque handle to communicator */
typedef struct mscclpp_ncclComm* mscclpp_ncclComm_t;
#define MSCCLPP_NCCL_COMM_NULL NULL

#define MSCCLPP_NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[MSCCLPP_NCCL_UNIQUE_ID_BYTES]; } mscclpp_ncclUniqueId;

/* Error type */
typedef enum { mscclpp_ncclSuccess                 =  0,
               mscclpp_ncclUnhandledCudaError      =  1,
               mscclpp_ncclSystemError             =  2,
               mscclpp_ncclInternalError           =  3,
               mscclpp_ncclInvalidArgument         =  4,
               mscclpp_ncclInvalidUsage            =  5,
               mscclpp_ncclRemoteError             =  6,
               mscclpp_ncclInProgress              =  7,
               mscclpp_ncclNumResults              =  8 } mscclpp_ncclResult_t;

#define MSCCLPP_NCCL_CONFIG_UNDEF_INT INT_MIN
#define MSCCLPP_NCCL_CONFIG_UNDEF_PTR NULL
#define MSCCLPP_NCCL_SPLIT_NOCOLOR -1

/* Communicator configuration. Users can assign value to attributes to specify the
 * behavior of a communicator. */
typedef struct mscclpp_ncclConfig_v21700 {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;
  /* attributes that users are able to customize. */
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  const char *netName;
  int splitShare;
} mscclpp_ncclConfig_t;

/* Config initializer must be assigned to initialize config structure when it is created.
 * Not initialized config will result in MSCCLPP_NCCL error. */
#define MSCCLPP_NCCL_CONFIG_INITIALIZER {                                       \
  sizeof(mscclpp_ncclConfig_t), /* size */                                      \
  0xcafebeef,           /* magic */                                     \
  MSCCLPP_NCCL_VERSION(MSCCLPP_NCCL_MAJOR, MSCCLPP_NCCL_MINOR, MSCCLPP_NCCL_PATCH), /* version */       \
  MSCCLPP_NCCL_CONFIG_UNDEF_INT,                    /* blocking */              \
  MSCCLPP_NCCL_CONFIG_UNDEF_INT,                    /* cgaClusterSize */        \
  MSCCLPP_NCCL_CONFIG_UNDEF_INT,                    /* minCTAs */               \
  MSCCLPP_NCCL_CONFIG_UNDEF_INT,                    /* maxCTAs */               \
  MSCCLPP_NCCL_CONFIG_UNDEF_PTR,                    /* netName */               \
  MSCCLPP_NCCL_CONFIG_UNDEF_INT                     /* splitShare */            \
}

/* Return the MSCCLPP_NCCL_VERSION_CODE of the MSCCLPP_NCCL library in the supplied integer.
 * This integer is coded with the MAJOR, MINOR and PATCH level of the
 * MSCCLPP_NCCL library
 */
mscclpp_ncclResult_t  mscclpp_ncclGetVersion(int *version);
mscclpp_ncclResult_t pmscclpp_ncclGetVersion(int *version);

/* Generates an Id to be used in mscclpp_ncclCommInitRank. mscclpp_ncclGetUniqueId should be
 * called once and the Id should be distributed to all ranks in the
 * communicator before calling mscclpp_ncclCommInitRank. */
mscclpp_ncclResult_t  mscclpp_ncclGetUniqueId(mscclpp_ncclUniqueId* uniqueId);
mscclpp_ncclResult_t pmscclpp_ncclGetUniqueId(mscclpp_ncclUniqueId* uniqueId);

/* Create a new communicator (multi thread/process version) with a configuration
 * set by users. */
mscclpp_ncclResult_t  mscclpp_ncclCommInitRankConfig(mscclpp_ncclComm_t* comm, int nranks, mscclpp_ncclUniqueId commId, int rank, mscclpp_ncclConfig_t* config);
mscclpp_ncclResult_t pmscclpp_ncclCommInitRankConfig(mscclpp_ncclComm_t* comm, int nranks, mscclpp_ncclUniqueId commId, int rank, mscclpp_ncclConfig_t* config);

/* Creates a new communicator (multi thread/process version).
 * rank must be between 0 and nranks-1 and unique within a communicator clique.
 * Each rank is associated to a CUDA device, which has to be set before calling
 * mscclpp_ncclCommInitRank.
 * mscclpp_ncclCommInitRank implicitly syncronizes with other ranks, so it must be
 * called by different threads/processes or use mscclpp_ncclGroupStart/mscclpp_ncclGroupEnd. */
mscclpp_ncclResult_t  mscclpp_ncclCommInitRank(mscclpp_ncclComm_t* comm, int nranks, mscclpp_ncclUniqueId commId, int rank);
mscclpp_ncclResult_t pmscclpp_ncclCommInitRank(mscclpp_ncclComm_t* comm, int nranks, mscclpp_ncclUniqueId commId, int rank);

/* Creates a clique of communicators (single process version).
 * This is a convenience function to create a single-process communicator clique.
 * Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(mscclpp_ncclComm_t).
 * If devlist is NULL, the first ndev CUDA devices are used.
 * Order of devlist defines user-order of processors within the communicator. */
mscclpp_ncclResult_t  mscclpp_ncclCommInitAll(mscclpp_ncclComm_t* comm, int ndev, const int* devlist);
mscclpp_ncclResult_t pmscclpp_ncclCommInitAll(mscclpp_ncclComm_t* comm, int ndev, const int* devlist);

/* Finalize a communicator. mscclpp_ncclCommFinalize flushes all issued communications,
 * and marks communicator state as mscclpp_ncclInProgress. The state will change to mscclpp_ncclSuccess
 * when the communicator is globally quiescent and related resources are freed; then,
 * calling mscclpp_ncclCommDestroy can locally free the rest of the resources (e.g. communicator
 * itself) without blocking. */
mscclpp_ncclResult_t  mscclpp_ncclCommFinalize(mscclpp_ncclComm_t comm);
mscclpp_ncclResult_t pmscclpp_ncclCommFinalize(mscclpp_ncclComm_t comm);

/* Frees local resources associated with communicator object. */
mscclpp_ncclResult_t  mscclpp_ncclCommDestroy(mscclpp_ncclComm_t comm);
mscclpp_ncclResult_t pmscclpp_ncclCommDestroy(mscclpp_ncclComm_t comm);

/* Frees resources associated with communicator object and aborts any operations
 * that might still be running on the device. */
mscclpp_ncclResult_t  mscclpp_ncclCommAbort(mscclpp_ncclComm_t comm);
mscclpp_ncclResult_t pmscclpp_ncclCommAbort(mscclpp_ncclComm_t comm);

/* Creates one or more communicators from an existing one.
 * Ranks with the same color will end up in the same communicator.
 * Within the new communicator, key will be used to order ranks.
 * MSCCLPP_NCCL_SPLIT_NOCOLOR as color will indicate the rank will not be part of any group
 * and will therefore return a NULL communicator.
 * If config is NULL, the new communicator will inherit the original communicator's
 * configuration*/
mscclpp_ncclResult_t  mscclpp_ncclCommSplit(mscclpp_ncclComm_t comm, int color, int key, mscclpp_ncclComm_t *newcomm, mscclpp_ncclConfig_t* config);
mscclpp_ncclResult_t pmscclpp_ncclCommSplit(mscclpp_ncclComm_t comm, int color, int key, mscclpp_ncclComm_t *newcomm, mscclpp_ncclConfig_t* config);

/* Returns a string for each error code. */
const char*  mscclpp_ncclGetErrorString(mscclpp_ncclResult_t result);
const char* pmscclpp_ncclGetErrorString(mscclpp_ncclResult_t result);

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
const char*  mscclpp_ncclGetLastError(mscclpp_ncclComm_t comm);
const char* pmscclpp_ncclGetLastError(mscclpp_ncclComm_t comm);

/* Checks whether the comm has encountered any asynchronous errors */
mscclpp_ncclResult_t  mscclpp_ncclCommGetAsyncError(mscclpp_ncclComm_t comm, mscclpp_ncclResult_t *asyncError);
mscclpp_ncclResult_t pmscclpp_ncclCommGetAsyncError(mscclpp_ncclComm_t comm, mscclpp_ncclResult_t *asyncError);

/* Gets the number of ranks in the communicator clique. */
mscclpp_ncclResult_t  mscclpp_ncclCommCount(const mscclpp_ncclComm_t comm, int* count);
mscclpp_ncclResult_t pmscclpp_ncclCommCount(const mscclpp_ncclComm_t comm, int* count);

/* Returns the cuda device number associated with the communicator. */
mscclpp_ncclResult_t  mscclpp_ncclCommCuDevice(const mscclpp_ncclComm_t comm, int* device);
mscclpp_ncclResult_t pmscclpp_ncclCommCuDevice(const mscclpp_ncclComm_t comm, int* device);

/* Returns the user-ordered "rank" associated with the communicator. */
mscclpp_ncclResult_t  mscclpp_ncclCommUserRank(const mscclpp_ncclComm_t comm, int* rank);
mscclpp_ncclResult_t pmscclpp_ncclCommUserRank(const mscclpp_ncclComm_t comm, int* rank);

/* Reduction operation selector */
typedef enum { mscclpp_ncclNumOps_dummy = 5 } mscclpp_ncclRedOp_dummy_t;
typedef enum { mscclpp_ncclSum        = 0,
               mscclpp_ncclProd       = 1,
               mscclpp_ncclMax        = 2,
               mscclpp_ncclMin        = 3,
               mscclpp_ncclAvg        = 4,
               /* mscclpp_ncclNumOps: The number of built-in mscclpp_ncclRedOp_t values. Also
                * serves as the least possible value for dynamic mscclpp_ncclRedOp_t's
                * as constructed by mscclpp_ncclRedOpCreate*** functions. */
               mscclpp_ncclNumOps     = 5,
               /* mscclpp_ncclMaxRedOp: The largest valid value for mscclpp_ncclRedOp_t.
                * It is defined to be the largest signed value (since compilers
                * are permitted to use signed enums) that won't grow
                * sizeof(mscclpp_ncclRedOp_t) when compared to previous MSCCLPP_NCCL versions to
                * maintain ABI compatibility. */
               mscclpp_ncclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(mscclpp_ncclRedOp_dummy_t))
             } mscclpp_ncclRedOp_t;

/* Data types */
typedef enum { mscclpp_ncclInt8       = 0, mscclpp_ncclChar       = 0,
               mscclpp_ncclUint8      = 1,
               mscclpp_ncclInt32      = 2, mscclpp_ncclInt        = 2,
               mscclpp_ncclUint32     = 3,
               mscclpp_ncclInt64      = 4,
               mscclpp_ncclUint64     = 5,
               mscclpp_ncclFloat16    = 6, mscclpp_ncclHalf       = 6,
               mscclpp_ncclFloat32    = 7, mscclpp_ncclFloat      = 7,
               mscclpp_ncclFloat64    = 8, mscclpp_ncclDouble     = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(__CUDA_FP8_TYPES_EXIST__)
               mscclpp_ncclBfloat16   = 9,
               mscclpp_ncclFp8E4M3    = 10,
               mscclpp_ncclFp8E5M2    = 11,
               mscclpp_ncclNumTypes   = 12
#elif defined(__CUDA_BF16_TYPES_EXIST__)
               mscclpp_ncclBfloat16   = 9,
               mscclpp_ncclNumTypes   = 10
#else
               mscclpp_ncclNumTypes   = 9
#endif
} mscclpp_ncclDataType_t;

/* mscclpp_ncclScalarResidence_t: Location and dereferencing logic for scalar arguments. */
typedef enum {
  /* mscclpp_ncclScalarDevice: The scalar is in device-visible memory and will be
   * dereferenced while the collective is running. */
  mscclpp_ncclScalarDevice = 0,

  /* mscclpp_ncclScalarHostImmediate: The scalar is in host-visible memory and will be
   * dereferenced before the mscclpp_ncclRedOpCreate***() function returns. */
  mscclpp_ncclScalarHostImmediate = 1
} mscclpp_ncclScalarResidence_t;

/*
 * mscclpp_ncclRedOpCreatePreMulSum
 *
 * Creates a new reduction operator which pre-multiplies input values by a given
 * scalar locally before reducing them with peer values via summation. For use
 * only with collectives launched against *comm* and *datatype*. The
 * *residence* argument indicates how/when the memory pointed to by *scalar*
 * will be dereferenced. Upon return, the newly created operator's handle
 * is stored in *op*.
 */
mscclpp_ncclResult_t  mscclpp_ncclRedOpCreatePreMulSum(mscclpp_ncclRedOp_t *op, void *scalar, mscclpp_ncclDataType_t datatype, mscclpp_ncclScalarResidence_t residence, mscclpp_ncclComm_t comm);
mscclpp_ncclResult_t pmscclpp_ncclRedOpCreatePreMulSum(mscclpp_ncclRedOp_t *op, void *scalar, mscclpp_ncclDataType_t datatype, mscclpp_ncclScalarResidence_t residence, mscclpp_ncclComm_t comm);

/*
 * mscclpp_ncclRedOpDestroy
 *
 * Destroys the reduction operator *op*. The operator must have been created by
 * mscclpp_ncclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
 * destroyed as soon as the last MSCCLPP_NCCL function which is given that operator returns.
 */
mscclpp_ncclResult_t mscclpp_ncclRedOpDestroy(mscclpp_ncclRedOp_t op, mscclpp_ncclComm_t comm);
mscclpp_ncclResult_t pmscclpp_ncclRedOpDestroy(mscclpp_ncclRedOp_t op, mscclpp_ncclComm_t comm);

/*
 * Collective communication operations
 *
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * They return when operations have been enqueued on the CUDA stream.
 *
 * Since they may perform inter-CPU synchronization, each call has to be done
 * from a different thread or process, or need to use Group Semantics (see
 * below).
 */

/*
 * Reduce
 *
 * Reduces data arrays of length count in sendbuff into recvbuff using op
 * operation.
 * recvbuff may be NULL on all calls except for root device.
 * root is the rank (not the CUDA device) where data will reside after the
 * operation is complete.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mscclpp_ncclResult_t  mscclpp_ncclReduce(const void* sendbuff, void* recvbuff, size_t count, mscclpp_ncclDataType_t datatype,
    mscclpp_ncclRedOp_t op, int root, mscclpp_ncclComm_t comm, cudaStream_t stream);
mscclpp_ncclResult_t pmscclpp_ncclReduce(const void* sendbuff, void* recvbuff, size_t count, mscclpp_ncclDataType_t datatype,
    mscclpp_ncclRedOp_t op, int root, mscclpp_ncclComm_t comm, cudaStream_t stream);

/*
 * (deprecated) Broadcast (in-place)
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * This operation is implicitly in place.
 */
mscclpp_ncclResult_t  mscclpp_ncclBcast(void* buff, size_t count, mscclpp_ncclDataType_t datatype, int root,
    mscclpp_ncclComm_t comm, cudaStream_t stream);
mscclpp_ncclResult_t pmscclpp_ncclBcast(void* buff, size_t count, mscclpp_ncclDataType_t datatype, int root,
    mscclpp_ncclComm_t comm, cudaStream_t stream);

/*
 * Broadcast
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mscclpp_ncclResult_t  mscclpp_ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, mscclpp_ncclDataType_t datatype, int root,
    mscclpp_ncclComm_t comm, cudaStream_t stream);
mscclpp_ncclResult_t pmscclpp_ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, mscclpp_ncclDataType_t datatype, int root,
    mscclpp_ncclComm_t comm, cudaStream_t stream);

/*
 * All-Reduce
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mscclpp_ncclResult_t  mscclpp_ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    mscclpp_ncclDataType_t datatype, mscclpp_ncclRedOp_t op, mscclpp_ncclComm_t comm, cudaStream_t stream);
mscclpp_ncclResult_t pmscclpp_ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    mscclpp_ncclDataType_t datatype, mscclpp_ncclRedOp_t op, mscclpp_ncclComm_t comm, cudaStream_t stream);

/*
 * Reduce-Scatter
 *
 * Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
mscclpp_ncclResult_t  mscclpp_ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, mscclpp_ncclDataType_t datatype, mscclpp_ncclRedOp_t op, mscclpp_ncclComm_t comm,
    cudaStream_t stream);
mscclpp_ncclResult_t pmscclpp_ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, mscclpp_ncclDataType_t datatype, mscclpp_ncclRedOp_t op, mscclpp_ncclComm_t comm,
    cudaStream_t stream);

/*
 * All-Gather
 *
 * Each device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
mscclpp_ncclResult_t  mscclpp_ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    mscclpp_ncclDataType_t datatype, mscclpp_ncclComm_t comm, cudaStream_t stream);
mscclpp_ncclResult_t pmscclpp_ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    mscclpp_ncclDataType_t datatype, mscclpp_ncclComm_t comm, cudaStream_t stream);

/*
 * Send
 *
 * Send data from sendbuff to rank peer.
 *
 * Rank peer needs to call mscclpp_ncclRecv with the same datatype and the same count from this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple mscclpp_ncclSend and mscclpp_ncclRecv operations
 * need to progress concurrently to complete, they must be fused within a mscclpp_ncclGroupStart/
 * mscclpp_ncclGroupEnd section.
 */
mscclpp_ncclResult_t  mscclpp_ncclSend(const void* sendbuff, size_t count, mscclpp_ncclDataType_t datatype, int peer,
    mscclpp_ncclComm_t comm, cudaStream_t stream);
mscclpp_ncclResult_t pmscclpp_ncclSend(const void* sendbuff, size_t count, mscclpp_ncclDataType_t datatype, int peer,
    mscclpp_ncclComm_t comm, cudaStream_t stream);

/*
 * Receive
 *
 * Receive data from rank peer into recvbuff.
 *
 * Rank peer needs to call mscclpp_ncclSend with the same datatype and the same count to this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple mscclpp_ncclSend and mscclpp_ncclRecv operations
 * need to progress concurrently to complete, they must be fused within a mscclpp_ncclGroupStart/
 * mscclpp_ncclGroupEnd section.
 */
mscclpp_ncclResult_t pmscclpp_ncclRecv(void* recvbuff, size_t count, mscclpp_ncclDataType_t datatype, int peer,
    mscclpp_ncclComm_t comm, cudaStream_t stream);
mscclpp_ncclResult_t  mscclpp_ncclRecv(void* recvbuff, size_t count, mscclpp_ncclDataType_t datatype, int peer,
    mscclpp_ncclComm_t comm, cudaStream_t stream);

/* All-To-All
 *
 * Device (i) send (j)th block of data to device (j) and be placed as (i)th
 * block. Each block for sending/receiving has count elements, which means
 * that recvbuff and sendbuff should have a size of nranks*count elements.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mscclpp_ncclResult_t  mscclpp_ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    mscclpp_ncclDataType_t datatype, mscclpp_ncclComm_t comm, cudaStream_t stream);
mscclpp_ncclResult_t pmscclpp_ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    mscclpp_ncclDataType_t datatype, mscclpp_ncclComm_t comm, cudaStream_t stream);
/*! @brief Opaque handle to MSCCL algorithm */
//typedef int mscclAlgoHandle_t;

/*! @brief MSCCL Load Algorithm
 *
 * @details Load MSCCL algorithm file specified in mscclAlgoFilePath and return
 * its handle via mscclAlgoHandle. This API is expected to be called by MSCCL
 * scheduler instead of end users.
 */
//mscclpp_ncclResult_t  mscclLoadAlgo(const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, int rank);
//mscclpp_ncclResult_t pmscclLoadAlgo(const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, int rank);

/*! @brief MSCCL Run Algorithm
 *
 * @details Run MSCCL algorithm specified by mscclAlgoHandle. The parameter
 * list merges all possible parameters required by different operations as this
 * is a general-purposed API. This API is expected to be called by MSCCL
 * scheduler instead of end users.
 */
//mscclpp_ncclResult_t  mscclRunAlgo(
//    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
//    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
//    size_t count, mscclpp_ncclDataType_t dataType, int root, int peer, mscclpp_ncclRedOp_t op,
//    mscclAlgoHandle_t mscclAlgoHandle, mscclpp_ncclComm_t comm, cudaStream_t stream);
//mscclpp_ncclResult_t pmscclRunAlgo(
//    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
//    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
//    size_t count, mscclpp_ncclDataType_t dataType, int root, int peer, mscclpp_ncclRedOp_t op,
//    mscclAlgoHandle_t mscclAlgoHandle, mscclpp_ncclComm_t comm, cudaStream_t stream);

/*! @brief MSCCL Load Algorithm
 *
 * @details Unload MSCCL algorithm previous loaded using its handle. This API
 * is expected to be called by MSCCL scheduler instead of end users.
 */
//mscclpp_ncclResult_t mscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle);
//mscclpp_ncclResult_t pmscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle);

/*
 * Group semantics
 *
 * When managing multiple GPUs from a single thread, and since MSCCLPP_NCCL collective
 * calls may perform inter-CPU synchronization, we need to "group" calls for
 * different ranks/devices into a single call.
 *
 * Grouping MSCCLPP_NCCL calls as being part of the same collective operation is done
 * using mscclpp_ncclGroupStart and mscclpp_ncclGroupEnd. mscclpp_ncclGroupStart will enqueue all
 * collective calls until the mscclpp_ncclGroupEnd call, which will wait for all calls
 * to be complete. Note that for collective communication, mscclpp_ncclGroupEnd only
 * guarantees that the operations are enqueued on the streams, not that
 * the operation is effectively done.
 *
 * Both collective communication and mscclpp_ncclCommInitRank can be used in conjunction
 * of mscclpp_ncclGroupStart/mscclpp_ncclGroupEnd, but not together.
 *
 * Group semantics also allow to fuse multiple operations on the same device
 * to improve performance (for aggregated collective calls), or to permit
 * concurrent progress of multiple send/receive operations.
 */

/*
 * Group Start
 *
 * Start a group call. All calls to MSCCLPP_NCCL until mscclpp_ncclGroupEnd will be fused into
 * a single MSCCLPP_NCCL operation. Nothing will be started on the CUDA stream until
 * mscclpp_ncclGroupEnd.
 */
mscclpp_ncclResult_t  mscclpp_ncclGroupStart();
mscclpp_ncclResult_t pmscclpp_ncclGroupStart();

/*
 * Group End
 *
 * End a group call. Start a fused MSCCLPP_NCCL operation consisting of all calls since
 * mscclpp_ncclGroupStart. Operations on the CUDA stream depending on the MSCCLPP_NCCL operations
 * need to be called after mscclpp_ncclGroupEnd.
 */
mscclpp_ncclResult_t  mscclpp_ncclGroupEnd();
mscclpp_ncclResult_t pmscclpp_ncclGroupEnd();

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
