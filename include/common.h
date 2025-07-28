#pragma once
#include <vector>
#include <string>
#include <cstdint>

enum Algorithm { AES, RC4 };
enum AESMode { ECB, CBC };

struct Config {
    Algorithm algorithm;
    AESMode mode;
    bool encrypt;
    std::string input_directory, output_directory, key_file;
    int gpu_threads, gpu_blocks;
    size_t key_bytes;
    std::vector<uint8_t> iv;

    bool parse(int argc, char** argv);
    void print_usage() const;
};

struct KeyBundle {
    std::vector<uint8_t> key;
    std::vector<uint8_t> roundKeys;
    void expand_round_keys(const Config& cfg);
    size_t roundKeyBytes() const { return roundKeys.size(); }
};

std::vector<uint8_t> hex_to_bytes(const std::string& hex);
std::vector<std::string> list_files(const std::string&);
bool read_file(const std::string&, std::vector<uint8_t>&);
void save_output(const std::string&, const std::vector<uint8_t>&, const Config&);
