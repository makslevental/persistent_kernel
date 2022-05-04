#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <cstdio>
#include <vector>


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

template<class FloatType>
struct SummaryStats {
    SummaryStats(){};
    /**
   * Compute summary statistics for values in v.
   *
   * Note we do not attempt to compute stats in a numerically stable manner.
   *
   * Note this will modify v by doing a partial sort to compute the median.
   */
    SummaryStats(std::vector<FloatType> &v) {
        const FloatType sum = std::accumulate(v.begin(), v.end(), 0.0);
        mean = sum / v.size();
        if (v.size() > 1) {
            FloatType sqsum = 0.0;
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
    FloatType mean = std::numeric_limits<FloatType>::quiet_NaN();
    FloatType stdev = std::numeric_limits<FloatType>::quiet_NaN();
    FloatType median = std::numeric_limits<FloatType>::quiet_NaN();
    FloatType min = std::numeric_limits<FloatType>::quiet_NaN();
    FloatType max = std::numeric_limits<FloatType>::quiet_NaN();
};


template<class FloatType>
inline std::ostream &operator<<(std::ostream &os, const SummaryStats<FloatType> &summary) {
    os << summary.mean << " | "
       << summary.median << " | "
       << summary.stdev << " | "
       << summary.min << " | "
       << summary.max << " | ";
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
    std::cout << "| Launch | " << SummaryStats(launch_times) << std::endl;
    std::cout << "| Signal | " << SummaryStats(times) << std::endl;
    cudaEventDestroy(e);
}

__global__ void EmptyKernel() {}

void profile_conventional_launch_overhead() {
    const int N = 100000;

    float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < N; i++) {

        cudaEventRecord(start, 0);
        EmptyKernel<<<1, 1>>>();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cumulative_time = cumulative_time + time;
    }
    printf("Kernel launch overhead time:  %3.5f ms \n", cumulative_time / N);
}


int main(int, char **) {
    profile_conventional_launch_overhead();

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
