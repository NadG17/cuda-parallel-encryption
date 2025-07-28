#include "key_management.h"
#include "common.h"
#include <fstream>
#include <random>

bool load_or_generate_key(const Config& cfg, KeyBundle& kb) {
    if (!cfg.key_file.empty()) {
        std::ifstream kf(cfg.key_file);
        if (!kf) return false;
        std::string hex;
        kf >> hex;
        kb.key = hex_to_bytes(hex);
    } else {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        kb.key.resize(cfg.key_bytes);
        for (auto &b : kb.key) b = rng() & 0xFF;
    }
    if (cfg.algorithm == AES) {
        kb.expand_round_keys(cfg);
    }
    return true;
}
