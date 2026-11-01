#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/kernels/iq_kernels.hpp"

namespace strata::core {
class WeightTable;

// C6 (44/45): an exl3-native pack's dense linears ride exl3_dense.txt as regions of dense.bin;
// weights.cpp synthesizes their metadata rows (WeightKind::Exl3, force-skipped: no canonical
// bytes), and this provider uploads each region and publishes the device views.  A published
// WeightRef carries native_type = 200 (kExl3Fmt), native_data = a stable kernels::Exl3DenseDesc*
// owned here (NOT a device pointer: every consumer checks native_type before touching native_data
// for 200), and native_q8_1 = one shared activation scratch (NativeDense's discipline).  Keep the
// provider alive until every graph that references these views is destroyed.
class Exl3Dense {
public:
    Exl3Dense() = default;
    ~Exl3Dense();
    Exl3Dense(const Exl3Dense&) = delete;
    Exl3Dense& operator=(const Exl3Dense&) = delete;

    /// Publish every Exl3-kind reference in `table`. A no-op (0 tensors) when none exist: the
    /// sidecar presence is already reflected by weights.cpp, so no separate sidecar read here.
    bool load(const std::string& pack_dir, WeightTable& table, std::string& err);

    size_t tensor_count() const { return descs_.size(); }
    uint64_t weight_bytes() const { return bytes_; }

private:
    std::vector<void*> allocs_;                     // one device allocation per region
    std::vector<std::unique_ptr<strata::kernels::Exl3DenseDesc>> descs_;
    void* scratch_ = nullptr;                       // shared q8_1 activation scratch
    uint64_t bytes_ = 0;
};

}  // namespace strata::core
