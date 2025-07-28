#include "file_processor.h"
#include "common.h"
#include <filesystem>
#include <fstream>

std::vector<std::string> list_files(const std::string& dir) {
    std::vector<std::string> files;
    for (auto& p : std::filesystem::recursive_directory_iterator(dir)) {
        if (p.is_regular_file()) files.push_back(p.path().string());
    }
    return files;
}

bool read_file(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    data.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

void save_output(const std::string& infile, const std::vector<uint8_t>& data, const Config& cfg) {
    std::string outname = cfg.output_directory + "/" + std::filesystem::path(infile).filename().string();
    outname += (cfg.encrypt ? ".enc" : ".dec");
    std::ofstream f(outname, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
}
