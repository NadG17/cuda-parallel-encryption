#pragma once
#include "common.h"
#include <chrono>
#include <cuda_runtime.h>

struct PerformanceMonitor {
    const Config &cfg;
    PerformanceMonitor(const Config&);
    void init_gpu();
    void log_session_start();
    void log_file(const std::string&, size_t bytes);
    void log_session_end(int totalFiles);
    void generate_report(const std::string& outdir);
private:
    cudaDeviceProp dev;
    std::chrono::high_resolution_clock::time_point start, end;
    size_t processed_bytes = 0;
    int file_count = 0;
};
