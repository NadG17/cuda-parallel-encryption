#pragma once
#include <vector>
#include <string>
bool read_file(const std::string&, std::vector<uint8_t>&);
void save_output(const std::string&, const std::vector<uint8_t>&, const Config&);
std::vector<std::string> list_files(const std::string&);
