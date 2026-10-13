// A weight that was repacked for the CPU must read back byte for byte. Build against the static ggml libraries of a Cale build:
//   c++ -std=c++17 -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src -I<llama.cpp>/ggml/src/ggml-cpu llama/tests/test_unrepack.cpp <libs> -o test_unrepack
// (scripts/check_unrepack.sh does this.) Prints one line per shape. Exit 1 on any difference.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "repack.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main() {
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_repack_buffer_type();
    int bad = 0;
    const int shapes[][2] = {{256, 64}, {2048, 512}, {512, 8}, {2048, 8192}};
    for (auto & sh : shapes) {
        const int64_t ne0 = sh[0], ne1 = sh[1];
        ggml_init_params ip = {ggml_tensor_overhead() * 4, nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ne0, ne1);
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> plain(nbytes), back(nbytes, 0);
        std::mt19937 rng(7);
        for (auto & b : plain) b = (uint8_t) rng();
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, nbytes + 4096);
        if (!buf) { printf("%lld x %lld: no buffer\n", (long long) ne0, (long long) ne1); return 1; }
        ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));
        ggml_backend_tensor_set(t, plain.data(), 0, nbytes);
        // The scheduler asks for whole tensors. A repacked tensor is not byte for byte the plain data: check that first.
        const bool changed = memcmp(t->data, plain.data(), nbytes) != 0;
        ggml_backend_tensor_get(t, back.data(), 0, nbytes);
        // A scale in the plain data may be any bit pattern: compare bytes, not floats.
        const bool same = memcmp(plain.data(), back.data(), nbytes) == 0;
        printf("%5lld x %-5lld  repacked-differs=%d  read-back-equal=%d\n", (long long) ne0, (long long) ne1, (int) changed, (int) same);
        if (!same) bad++;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    return bad ? 1 : 0;
}
