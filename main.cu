#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include <vector>

//#define USECPSEC 1000000ULL
//
// unsigned long long dtime_usec(unsigned long long start) {
//  timeval tv;
//  gettimeofday(&tv, 0);
//  return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
//}
//
//__global__ void tkernel() {}

// void profile_overhead() {
//   float dt_uspec = 0;
//   for (int i = 0; i < 1000; i++) {
//     tkernel<<<2000, 32>>>();
//     cudaDeviceSynchronize();
//     unsigned long long dt = dtime_usec(0);
//     unsigned long long dt1 = dt;
//     tkernel<<<2000, 32>>>();
//     dt = dtime_usec(dt);
//     cudaDeviceSynchronize();
//     dt1 = dtime_usec(dt1);
//     dt_uspec += dt / (float)USECPSEC;
//   }
//   printf("kernel launch: %fs", dt_uspec / 1000);
// }

inline double get_time() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch())
            .count();
}
__global__ void spin_wait_kernel(int32_t wait_value,
                                 volatile int32_t *wait_mem) {
    for (;;) {
        __threadfence_system();
        int32_t value = *wait_mem;
        if (value == wait_value)
            break;
    }
}

void launch_wait_kernel_spin(cudaStream_t stream, int32_t wait_value,
                             volatile int32_t *wait_mem) {
    spin_wait_kernel<<<1, 1, 0, stream>>>(wait_value, wait_mem);
}

void launch_wait_kernel_value(cudaStream_t stream, int32_t wait_value,
                              CUdeviceptr wait_mem) {
    cuStreamWaitValue32(stream, wait_mem, wait_value, CU_STREAM_WAIT_VALUE_EQ);
}

class Wait {
public:
    Wait() {
        cudaMallocHost(&wait_sync, sizeof(int32_t));
        __atomic_store_n(wait_sync, 0, __ATOMIC_SEQ_CST);
    }
    ~Wait() { cudaFreeHost(wait_sync); }
    virtual void wait(cudaStream_t stream) = 0;
    virtual void signal() { __atomic_store_n(wait_sync, 1, __ATOMIC_SEQ_CST); }

    int32_t *wait_sync __attribute__((aligned(64)));
};

class StreamOpWait : public Wait {
public:
    StreamOpWait() : Wait() { cuMemHostGetDevicePointer(&dev_ptr, wait_sync, 0); }
    ~StreamOpWait() {}
    void wait(cudaStream_t stream) override {
        launch_wait_kernel_value(stream, 1, dev_ptr);
    }
    CUdeviceptr dev_ptr;
};

class KernelWait : public Wait {
public:
    KernelWait() : Wait() { cudaHostGetDevicePointer(&dev_ptr, wait_sync, 0); }
    ~KernelWait() {}
    void wait(cudaStream_t stream) override {
        launch_wait_kernel_spin(stream, 1, dev_ptr);
    }
    int32_t *dev_ptr __attribute__((aligned(64)));
};

struct SummaryStats {
    SummaryStats(){};
    /**
   * Compute summary statistics for values in v.
   *
   * Note we do not attempt to compute stats in a numerically stable manner.
   *
   * Note this will modify v by doing a partial sort to compute the median.
   */
    SummaryStats(std::vector<double> &v) {
        const double sum = std::accumulate(v.begin(), v.end(), 0.0);
        mean = sum / v.size();
        if (v.size() > 1) {
            double sqsum = 0.0;
            for (const auto &x: v) {
                sqsum += (x - mean) * (x - mean);
            }
            stdev = std::sqrt(1.0 / (v.size() - 1) * sqsum);
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            // This is not correct for even-length vectors, but to quote
            // Numerical Recipes: "... formalists be damned".
            median = v[v.size() / 2];
            auto minmax = std::minmax_element(v.begin(), v.end());
            min = *minmax.first;
            max = *minmax.second;
        }
    }
    double mean = std::numeric_limits<double>::quiet_NaN();
    double stdev = std::numeric_limits<double>::quiet_NaN();
    double median = std::numeric_limits<double>::quiet_NaN();
    double min = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
};

inline std::ostream &operator<<(std::ostream &os, const SummaryStats &summary) {
    os << "mean " << summary.mean << "s "
       << "median " << summary.median << "s "
       << "stdev " << summary.stdev << "s "
       << "min " << summary.min << "s "
       << "max " << summary.max << "s ";
    return os;
}

void do_benchmark(cudaStream_t stream, Wait &wait) {
    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    std::vector<double> times, launch_times;
    for (int i = 0; i < 100000; ++i) {
        cudaStreamSynchronize(stream);

        double launch_start = get_time();
        wait.wait(stream);
        double launch_end = get_time();
        launch_times.push_back(launch_end - launch_start);

        cudaEventRecord(e, stream);
        double start = get_time();
        wait.signal();
        while (cudaEventQuery(e) == cudaErrorNotReady) {
        }
        double end = get_time();
        times.push_back(end - start);

        cudaStreamSynchronize(stream);
    }
    std::cout << "Launch wait: " << SummaryStats(launch_times) << std::endl;
    std::cout << "Signal end wait: " << SummaryStats(times) << std::endl;
    cudaEventDestroy(e);
}

int main(int, char **) {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    {
        StreamOpWait stream_op_wait;
        std::cout << "StreamOp wait:" << std::endl;
        do_benchmark(stream, stream_op_wait);

        KernelWait kernel_wait;
        std::cout << "Kernel wait:" << std::endl;
        do_benchmark(stream, kernel_wait);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return 0;
}
