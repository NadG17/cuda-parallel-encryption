#pragma once
#include <vector>
#include "common.h"
void aes_encrypt_gpu(const std::vector<uint8_t>&, std::vector<uint8_t>&, const KeyBundle&, const Config&);
