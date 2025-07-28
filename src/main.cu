#include <iostream>
#include <vector>
#include <string>
#include "common.h"
#include "aes.h"
#include "rc4.h"
#include "file_processor.h"
#include "key_management.h"
#include "performance_monitor.h"

int main(int argc, char** argv) {
    Config cfg;
    if (!cfg.parse(argc, argv)) {
        cfg.print_usage();
        return EXIT_FAILURE;
    }

    KeyBundle keyBundle;
    if (!load_or_generate_key(cfg, keyBundle)) {
        std::cerr << "Key loading/generation failed\n";
        return EXIT_FAILURE;
    }

    std::vector<std::string> inputFiles = list_files(cfg.input_directory);
    PerformanceMonitor monitor(cfg);
    monitor.init_gpu();

    monitor.log_session_start();

    for (auto &filename : inputFiles) {
        std::vector<uint8_t> plaintext;
        if (!read_file(filename, plaintext)) continue;
        std::vector<uint8_t> ciphertext(plaintext.size());
        if (cfg.algorithm == AES) {
            aes_encrypt_gpu(plaintext, ciphertext, keyBundle, cfg);
        } else {
            rc4_encrypt_gpu(plaintext, ciphertext, keyBundle);
        }
        save_output(filename, ciphertext, cfg);
        monitor.log_file(filename, plaintext.size());
    }

    monitor.log_session_end(inputFiles.size());
    monitor.generate_report(cfg.output_directory);

    return EXIT_SUCCESS;
}
