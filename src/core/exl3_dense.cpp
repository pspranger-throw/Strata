// C6 (44/45): upload + publish for exl3-native packs' dense linears.  The metadata rows already
// exist (weights.cpp synthesized them from exl3_dense.txt and force-skipped them); this file owns
// the device side: one allocation per region, one shared q8_1 scratch, one stable descriptor per
// tensor behind WeightRef.native_data.  K is not stored in the row: it is recovered from the exact
// region span by the pack's own algebra - the same equation the packer, the v6 loader, the adapter
// and G0 all agree on (44's four-way table) - so a corrupted span is refused, never guessed at.
#include "strata/core/exl3_dense.hpp"

#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::core {
namespace {
constexpr uint64_t kChunk = 64ull << 20;   // pinned staging, as in WeightTable::load
}

Exl3Dense::~Exl3Dense() {
    for (void* p : allocs_)
        if (p) cudaFree(p);
    if (scratch_) cudaFree(scratch_);
}

bool Exl3Dense::load(const std::string& pack_dir, WeightTable& table, std::string& err) {
    if (!descs_.empty()) { err = "exl3 dense: already loaded"; return false; }

    std::vector<std::tuple<uint64_t, std::string, WeightRef*>> refs;
    for (auto& entry : table.table_)
        if (entry.second.kind == WeightKind::Exl3) refs.emplace_back(entry.second.src_off, entry.first, &entry.second);
    if (refs.empty()) return true;                       // not an exl3 pack: nothing to do
    std::sort(refs.begin(), refs.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    const std::string bin = pack_dir + "/dense.bin";
    struct stat st{};
    if (stat(bin.c_str(), &st) != 0) { err = "exl3 dense: cannot stat " + bin; return false; }
    const uint64_t fsize = (uint64_t) st.st_size;
    const int fd = open(bin.c_str(), O_RDONLY);
    if (fd < 0) { err = "exl3 dense: cannot open " + bin; return false; }
    void* stage = nullptr;
    if (cudaHostAlloc(&stage, kChunk, cudaHostAllocDefault) != cudaSuccess) {
        close(fd);
        err = "exl3 dense: pinned staging failed";
        return false;
    }

    // All validation and allocation completes before any reference is published (NativeDense's
    // discipline): a half-attached exl3 tensor is a wrong weight, and a wrong weight is plausible.
    size_t max_in = 0;
    for (auto& entry : refs) {
        WeightRef* wr = std::get<2>(entry);
        const std::string& nm = std::get<1>(entry);
        const uint64_t k = (uint64_t) wr->ne0, n = (uint64_t) wr->ne1;
        const uint64_t heads = 2 * (k + n);
        const uint64_t per_bit = (k / 16) * (n / 16) * 32;
        if (k == 0 || n == 0 || k % 128 != 0 || n % 128 != 0 || !per_bit || wr->src_bytes <= heads ||
            (wr->src_bytes - heads) % per_bit != 0) {
            cudaFreeHost(stage);
            close(fd);
            err = "exl3 dense: " + nm + " does not match the exl3 region algebra";
            return false;
        }
        const int K = (int) ((wr->src_bytes - heads) / per_bit);
        if (K < 1 || K > 8) {
            cudaFreeHost(stage);
            close(fd);
            err = "exl3 dense: " + nm + " recovers K = " + std::to_string(K);
            return false;
        }
        if (wr->src_off > fsize || wr->src_bytes > fsize - wr->src_off) {
            cudaFreeHost(stage);
            close(fd);
            err = "exl3 dense: truncated region for " + nm;
            return false;
        }
        void* dev = nullptr;
        if (cudaMalloc(&dev, wr->src_bytes) != cudaSuccess) {
            cudaFreeHost(stage);
            close(fd);
            err = "exl3 dense: cudaMalloc failed for " + nm;
            return false;
        }
        allocs_.push_back(dev);
        bool ok = true;
        for (uint64_t done = 0; done < wr->src_bytes && ok; ) {
            const uint64_t want = std::min(kChunk, wr->src_bytes - done);
            const ssize_t rd = pread(fd, stage, (size_t) want, (off_t) (wr->src_off + done));
            ok = (uint64_t) rd == want &&
                 cudaMemcpy((char*) dev + done, stage, (size_t) want, cudaMemcpyHostToDevice) == cudaSuccess;
            done += want;
        }
        if (!ok) {
            cudaFreeHost(stage);
            close(fd);
            err = "exl3 dense: upload failed for " + nm;
            return false;
        }
        auto d = std::make_unique<strata::kernels::Exl3DenseDesc>();
        d->base = dev;
        d->k = (int) k;
        d->n = (int) n;
        d->K = K;
        d->cb = 2;                     // mul1: the sidecar header and the G0 census allow nothing else
        descs_.push_back(std::move(d));
        max_in = std::max(max_in, (size_t) k);
        bytes_ += wr->src_bytes;
    }
    cudaFreeHost(stage);
    close(fd);

    if (cudaMalloc(&scratch_, strata::kernels::native_q8_1_bytes(max_in)) != cudaSuccess) {
        err = "exl3 dense: activation scratch failed";
        return false;
    }
    for (auto& entry : refs) {
        std::get<2>(entry)->native_type = strata::kernels::cpu::kExl3Fmt;
        std::get<2>(entry)->native_q8_1 = scratch_;
    }
    // descs_ and refs were built in the same ascending src_off order, one entry each: pair them.
    for (size_t i = 0; i < refs.size(); ++i) std::get<2>(refs[i])->native_data = descs_[i].get();
    return true;
}

}  // namespace strata::core
