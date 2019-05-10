/**
 * This file contains custom Python-C++ bridge interfaces for accessing CUPTI.
 * The interfaces can be accessed by running `from utils import cupti` in
 * Python, after building the cupti.so file (see build_cupti.sh).
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda.h>
#include <cupti.h>

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

// size of a single activity record, in bytes
// this only applies for KIND_KERNEL, and may be different for other types
#define SINGLE_RECORD_SIZE 144

// number of activity records to put in a single activity buffer
#define MAX_NUM_RECORDS_PER_BUFFER 128

// size of buffer to give when requested by CUPTI Activity API
#define BUFFER_SIZE (SINGLE_RECORD_SIZE * MAX_NUM_RECORDS_PER_BUFFER)

#define DRIVER_API_CALL(apiFuncCall)                                           \
  do {                                                                         \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
      const char *errstr;                                                      \
      cuGetErrorName(_status, &errstr);                                        \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",     \
              __FILE__, __LINE__, #apiFuncCall, errstr);                       \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define CUPTI_CALL(call)                                                       \
  do {                                                                         \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
      const char *errstr;                                                      \
      cuptiGetResultString(_status, &errstr);                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",     \
              __FILE__, __LINE__, #call, errstr);                              \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)


/**
 * Internal functions that are not directly exposed to Python.
 */

// data structure for storing start and end timestamps for kernels
typedef std::vector<std::tuple<std::string,
                              unsigned long long,
                              unsigned long long>> timelog;

static timelog kernelTimestamps;

// fetch kernel name and timestamps from record and put them in timelog
static void
recordActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *) record;
      kernelTimestamps.push_back(std::make_tuple(std::string(kernel->name),
                                                 (unsigned long long)kernel->start,
                                                 (unsigned long long)kernel->end));
      break;
    }
  default:
    // we shouldn't get any other activity types,
    // because we only asked for KIND_KERNEL
    break;
  }
}

// hand over a heap buffer of BUFFER_SIZE to CUPTI
static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUFFER_SIZE);
  if (bfr == NULL) {
    printf("[ERROR] out of memory\n");
    exit(-1);
  }

  *size = BUFFER_SIZE;
  *buffer = bfr;
  // put no limit to the number of records, so CUPTI fills the buffer as much
  // as the buffer size allows
  *maxNumRecords = 0;
}

// iterate through the buffer to collect info from each record
static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        recordActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      }
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      // this should not happen, because we gave *maxNumRecords=0
      printf("[WARNING] Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}


/**
 * Functions that are exposed to the Python level.
 */

// initialization methods that are required to use CUPTI Activity API
void initialize() {
  // we are only interested in kernel executions
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested,
                                            bufferCompleted));

  // pre-allocate memory for the first MAX_NUM_RECORDS_PER_BUFFER records
  // note that this is not a hard limit; kernelTimestamps is a vector
  kernelTimestamps.reserve(MAX_NUM_RECORDS_PER_BUFFER);
}

// flush out any remaining records so that bufferCompleted() is invoked
void flush() {
  cuptiActivityFlushAll(0);
}

// return the collected statistics in the form of a list of Python tuples
// [(kernelName1, kernelStartTimestamp1, kernelEndTimestamp1), ...]
timelog report() {
  // we create a copy of kernelTimestamps and return the copy so that
  // we can empty out the contents of kernelTimestamps 
  timelog ret = timelog(kernelTimestamps);
  kernelTimestamps.clear();
  return ret;
}


namespace py = pybind11;

PYBIND11_MODULE(cupti, m) {
  m.def("initialize", &initialize);
  m.def("flush", &flush);
  m.def("report", &report);
}
