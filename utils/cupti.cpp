/**
 * This file contains custom Python-C++ bridge interfaces for accessing CUPTI.
 * The interfaces can be accessed by running `from utils import cupti` in
 * Python, after building the cupti.so file (see build.sh).
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda.h>
#include <cupti.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <string>

// assume that no more than MAX_NUM_KERNELS kernels will
// be run between two calls to KernelTracker.reset()
#define MAX_NUM_KERNELS 1000 

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


// forward declaration, for use in KernelTracker's constructor
void CUPTIAPI
callback(void *userdata, CUpti_CallbackDomain domain,
         CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);


/**
 * Record the names of all GPU kernels by adding a callback for kernel launches.
 *
 * This class assumes that a CUDA context has already been created and is
 * fetchable with cuCtxGetCurrent.
 * This class may not work with multi-thread or multi-process programs.
 */
class KernelTracker {
public:
  KernelTracker() {
    // We could do cuCtxCreate here and create our own context, but it seems
    // that there is no clear way to pass the context back to Python.
    // So instead, we just assume that a context is already available.
    DRIVER_API_CALL(cuCtxGetCurrent(&_context));

    // attach the callback function, and pass
    // a pointer to this class instance as user data
    CUPTI_CALL(cuptiSubscribe(&_subscriber, (CUpti_CallbackFunc)callback,
                              this));

    // we are only interested in kernel launches
    CUPTI_CALL(cuptiEnableCallback(1, _subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
  }

  ~KernelTracker() {
    CUPTI_CALL(cuptiUnsubscribe(_subscriber));
  }

  void record(const char *name) {
    const int curr = _counter++;
    if (curr < MAX_NUM_KERNELS) {
      // we only keep track of kernels up to MAX_NUM_KERNELS
      _kernelNames[curr] = std::string(name);
    } else if (curr == MAX_NUM_KERNELS) {
      fprintf(stderr, "[WARNING] Too many kernels. Will only track "
                      "the first %d kernels.\n", MAX_NUM_KERNELS);
    }
  }

  void reset() {
    _counter = 0;
  }

  std::vector<std::string> getKernelNames() {
    std::vector<std::string> ret;

    // convert atomic<int> into int so that we can use the ternary operator
    int num_kernels = _counter;
    num_kernels = num_kernels < MAX_NUM_KERNELS ? num_kernels : MAX_NUM_KERNELS;
    for (int i = 0; i < num_kernels; ++i) {
      ret.push_back(_kernelNames[i]);
    }
    return ret;
  }


private:
  CUcontext _context = 0;
  CUpti_SubscriberHandle _subscriber;

  // multiple callbacks can occur concurrently
  std::atomic_int _counter{0}; 
  // use array instead of vector to avoid runtime reallocations (performance)
  std::array<std::string, MAX_NUM_KERNELS> _kernelNames;
};


void CUPTIAPI
callback(void *userdata, CUpti_CallbackDomain domain,
         CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
  if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
    // ignore everything that is not a kernel launch
    return;
  }

  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    // we are only interested in CUPTI_API_ENTER
    return;
  }

  KernelTracker *cupti = (KernelTracker*)userdata;
  cupti->record(cbInfo->symbolName); // name of this kernel
}


namespace py = pybind11;

PYBIND11_MODULE(cupti, m) {
  py::class_<KernelTracker>(m, "KernelTracker")
      .def(py::init<>())
      .def("get_kernel_names", &KernelTracker::getKernelNames)
      .def("reset", &KernelTracker::reset);
}
