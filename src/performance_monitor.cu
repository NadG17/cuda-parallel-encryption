#include "performance_monitor.h"
#include <iostream>
#include <chrono>
#include <fstream>

PerformanceMonitor::PerformanceMonitor(const Config& cfg): cfg(cfg) {}

void PerformanceMonitor::init_gpu() {
    cudaGetDeviceProperties(&dev, 0);
}

void PerformanceMonitor::log_session_start() {
    start = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::log_file(const std::string& name, size_t bytes) {
    processed_bytes += bytes;
    file_count++;
}

void PerformanceMonitor::log_session_end(int totalFiles) {
    end = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::generate_report(const std::string& outdir) {
    double secs = std::chrono::duration<double>(end-start).count();
    double gb = processed_bytes / 1e9;
    double throughput = gb / secs;

    std::cout << "Session complete: " << file_count << " files, total " << gb << " GB in "
              << secs << " s → " << throughput << " GB/s\n";

    std::ofstream f(outdir + "/performance_report.txt");
    f << "GPU Encryption Performance Report\n";
    f << "===============================\n";
    f << "Total data processed: " << gb << " GB\n";
    f << "Time: " << secs << " s\n";
    f << "Throughput: " << throughput << " GB/s\n";
    f << "Device: " << dev.name << "\n";
}
