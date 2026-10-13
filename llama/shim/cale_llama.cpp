// A small, stable C interface over the patched llama.cpp. Python calls this through ctypes.
// Why a shim: llama.cpp's own structs change between versions. This file is built together with the
// pinned llama.cpp (see llama/build.sh), so the two always match.
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#ifdef CALE_BACKENDS_DL
#  include "fit.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <psapi.h>
#  define CALE_API extern "C" __declspec(dllexport)
#else
#  include <dlfcn.h>
#  include <sys/mman.h>
#  include <unistd.h>
#  define CALE_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
struct cale_slot {
    int32_t                  state = 0;            // 0 free, 1 reading the prompt, 2 a token is chosen and waits to be read, 3 waiting for the caller to choose
    std::vector<llama_token> kv;                   // tokens of this answer's sequence in the memory
    std::vector<llama_token> prompt;               // the whole prompt of the answer in progress
    size_t                   read = 0;             // how much of the prompt is read
    llama_token              pending = -1;
    llama_sampler *          smpl = nullptr;
    bool                     guided = false;       // the caller chooses the tokens (see cale_slot_topk)
    int32_t                  last_idx = -1;        // where this slot's logits are in the last batch
    int32_t                  reused = 0;
};

struct cale_ctx {
    llama_model *       model  = nullptr;
    llama_context *     ctx    = nullptr;
    const llama_vocab * vocab  = nullptr;
    llama_sampler *     smpl   = nullptr;
    int32_t             n_ctx  = 0;
    int32_t             n_batch = 512;
    int32_t             n_past = 0;
    llama_token         pending = -1;
    gguf_context *      gguf   = nullptr;      // the file's tensor table, to read a tensor back
    std::string         path;
    bool                repacked = false;      // weights were repacked into ordinary memory (not mapped from the file)
    bool                split = false;         // the file stores each down matrix as two halves
    std::string         placement;             // which device and layers this load used (for status and replies)
    std::vector<llama_token> kv_tokens;        // the tokens the key/value memory holds now, in order (prompt cache)
    std::vector<cale_slot> slots;              // answers that run together (cale_batch_step). The single-answer calls above use sequence 0 on their own.
    int32_t             reused = 0;            // how many tokens of the last prompt were taken from the memory instead of being read again
    std::atomic<int>    cancel{0};             // set from another thread: stop the work in progress
};

thread_local std::string g_err;
bool g_backend_ready = false;
std::vector<ggml_backend_dev_t> g_gpus;   // compute devices that are not the CPU, in the order cale_devices() lists them

std::string shim_dir() {
#if defined(_WIN32)
    HMODULE m = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) &shim_dir, &m);
    char buf[4096];
    const DWORD n = GetModuleFileNameA(m, buf, sizeof(buf));
    const std::string p(buf, n);
#else
    Dl_info info;
    const std::string p = (dladdr((void *) &shim_dir, &info) && info.dli_fname) ? info.dli_fname : "";
#endif
    const size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? "." : p.substr(0, k);
}

void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

#if defined(_WIN32)
// Debug aid, off unless CALE_CRASH_TRACE=1: print where an access violation happens (module and offset, then resolve the
// offset with `nm` or `addr2line` on the unstripped DLL). Windows hides native crashes inside Python's ctypes otherwise.
LONG WINAPI crash_trace(EXCEPTION_POINTERS * ep) {
    static std::atomic<int> once{0};
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || once.exchange(1)) return EXCEPTION_CONTINUE_SEARCH;
    fprintf(stderr, "CRASH_TRACE access violation: instruction %p, kind %d, address %p\n", ep->ExceptionRecord->ExceptionAddress,
            (int) ep->ExceptionRecord->ExceptionInformation[0], (void *) ep->ExceptionRecord->ExceptionInformation[1]);
    // The faulting thread's stack: every value that points into a loaded module's code is a candidate return address.
#if defined(_M_ARM64) || defined(__aarch64__)
    const uintptr_t * sp = (const uintptr_t *) ep->ContextRecord->Sp;
#else
    const uintptr_t * sp = (const uintptr_t *) ep->ContextRecord->Rsp;
#endif
    int shown = 0;
    for (int i = 0; i < 400 && shown < 40; ++i) {
        HMODULE m = nullptr;
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) sp[i], &m)) continue;
        char name[MAX_PATH] = "?";
        GetModuleFileNameA(m, name, sizeof(name));
        const char * base = strrchr(name, '\\');
        fprintf(stderr, "CRASH_TRACE stack[%d] %s +0x%llx\n", i, base ? base + 1 : name, (unsigned long long) (sp[i] - (uintptr_t) m));
        ++shown;
    }
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

void init_backends() {
    if (g_backend_ready) return;
#if defined(_WIN32)
    if (getenv("CALE_CRASH_TRACE")) AddVectoredExceptionHandler(1, crash_trace);
#endif
    llama_log_set(quiet_log, nullptr);
#ifdef CALE_BACKENDS_DL
    // Backends are separate libraries. Look next to this library, or where CALE_BACKEND_DIR says.
    const char * env = getenv("CALE_BACKEND_DIR");
    const std::string dir = (env && *env) ? std::string(env) : shim_dir();
    ggml_backend_load_all_from_path(dir.c_str());
#endif
    llama_backend_init();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) g_gpus.push_back(d);
    }
    g_backend_ready = true;
}

bool decode_tokens(cale_ctx * h, llama_token * t, int32_t n) {
    while (n > 0) {
        const int32_t k = n < h->n_batch ? n : h->n_batch;
        if (h->cancel.load() != 0) { g_err = "cancelled"; return false; }
        if (llama_decode(h->ctx, llama_batch_get_one(t, k)) != 0) {
            g_err = h->cancel.load() != 0 ? "cancelled" : "llama_decode failed";
            h->kv_tokens.clear();                                    // the memory may be half written: do not trust it
            llama_memory_clear(llama_get_memory(h->ctx), true);
            h->n_past = 0;
            return false;
        }
        h->kv_tokens.insert(h->kv_tokens.end(), t, t + k);
        h->n_past += k;
        t += k;
        n -= k;
    }
    return true;
}
}  // namespace

CALE_API const char * cale_last_error(void) { return g_err.c_str(); }

// One line per compute device that is not the CPU: "index|type|name|description|free_bytes|total_bytes".
// type is GPU (own memory), IGPU (shares the computer's memory) or ACCEL. The index is only a handle for cale_load_dev
// in this process. Pick a device by its type, name and memory, never by its index: the order changes with drivers.
CALE_API int32_t cale_devices(char * out, int32_t cap) {
    init_backends();
    std::string text;
    for (size_t i = 0; i < g_gpus.size(); ++i) {
        ggml_backend_dev_props pr;
        ggml_backend_dev_get_props(g_gpus[i], &pr);
        const char * type = pr.type == GGML_BACKEND_DEVICE_TYPE_IGPU ? "IGPU" : pr.type == GGML_BACKEND_DEVICE_TYPE_GPU ? "GPU" : "ACCEL";
        text += std::to_string(i) + "|" + type + "|" + pr.name + "|" + pr.description + "|" +
                std::to_string((unsigned long long) pr.memory_free) + "|" + std::to_string((unsigned long long) pr.memory_total) + "\n";
    }
    if ((int32_t) text.size() + 1 > cap) return -1;
    memcpy(out, text.c_str(), text.size() + 1);
    return (int32_t) text.size();
}

// Load a GGUF.
// repack: 0 = repack the weights when the file is in the Cale layout (see llama/prepare_gguf.py), 1 = always, 2 = never.
// Repacked weights are faster on the CPU but live in ordinary memory. Without repacking the file is memory-mapped,
// and only the parts a size touches are read from disk.
// gpu: -1 = CPU only, else the handle from cale_devices(). n_gpu_layers: -1 = let llama.cpp's own fit choose how much
// goes to the device (it leaves fit_margin_mb free), else that many layers.
// no_host: 1 = keep the weights that stay on the CPU in ordinary memory so they can be repacked. With a GPU loaded
// the default puts them in a pinned buffer, which is never repacked and made answers 2x slower (measured 2026-10-13).
CALE_API cale_ctx * cale_load_dev(const char * path, int32_t n_threads, int32_t n_ctx, int32_t repack,
                                  int32_t gpu, int32_t n_gpu_layers, int32_t no_host, int32_t fit_margin_mb) {
    init_backends();
    if (gpu >= (int32_t) g_gpus.size()) { g_err = "no such compute device"; return nullptr; }
    gguf_init_params gp = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) {
        g_err = std::string("not a GGUF file: ") + path;
        return nullptr;
    }
    const int64_t split_key = gguf_find_key(g, "cale.ffn_down_split");
    const bool split = split_key >= 0 && gguf_get_val_u32(g, split_key) == 1;
    const bool want_repack = repack == 1 || (repack == 0 && split);
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = {gpu >= 0 ? g_gpus[gpu] : nullptr, nullptr};
    mp.devices         = devs;            // an empty list means the CPU alone: a GPU that is present must not change a CPU load
    mp.use_extra_bufts = want_repack;     // off: a size is a view of the stock tensors and the file stays memory-mapped
    mp.no_host         = no_host != 0;
    if (gpu < 0) mp.n_gpu_layers = 0;
    else if (n_gpu_layers >= 0) mp.n_gpu_layers = n_gpu_layers;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    const char * ns = getenv("CALE_SLOTS");                       // answers that may run together (1 = one at a time)
    const int32_t n_slots = ns ? std::max(1, std::min(16, atoi(ns))) : 1;
    cp.n_seq_max = (uint32_t) n_slots;
    cp.kv_unified = true;                                         // all answers share one memory pool: the context length is not multiplied
    std::string placement = gpu < 0 ? "device=cpu" : "device=" + std::string(ggml_backend_dev_name(g_gpus[gpu]));
#ifdef CALE_BACKENDS_DL
    std::vector<float> tensor_split(llama_max_devices(), 0.0f);
    std::vector<llama_model_tensor_buft_override> overrides(llama_max_tensor_buft_overrides());
    if (gpu >= 0 && n_gpu_layers < 0) {
        std::vector<size_t> margins(llama_max_devices(), (size_t) (fit_margin_mb < 0 ? 0 : fit_margin_mb) * 1024 * 1024);
        const common_params_fit_status st = common_fit_params(path, &mp, &cp, tensor_split.data(), overrides.data(),
                                                              margins.data(), 512, nullptr, GGML_LOG_LEVEL_ERROR);
        if (st != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            gguf_free(g);
            g_err = "this model does not fit the device";
            return nullptr;
        }
        placement += ";fit=1";
    }
#else
    (void) fit_margin_mb;
    if (gpu >= 0) { gguf_free(g); g_err = "this build has no GPU backend"; return nullptr; }
#endif
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        gguf_free(g);
        g_err = std::string("could not load model: ") + path;
        return nullptr;
    }
    placement += ";gpu_layers=" + std::to_string(mp.n_gpu_layers) + ";no_host=" + std::to_string(no_host != 0);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        gguf_free(g);
        g_err = "could not create context";
        return nullptr;
    }
    auto * h = new cale_ctx();
    h->gguf = g;
    h->path = path;
    h->repacked = want_repack;
    h->split = split;
    h->model = model;
    h->ctx = ctx;
    h->placement = placement;
    h->slots.resize(n_slots);
    h->vocab = llama_model_get_vocab(model);
    h->n_ctx = (int32_t) llama_n_ctx(ctx);
    h->n_batch = (int32_t) llama_n_batch(ctx);
    llama_set_abort_callback(ctx, [](void * d) -> bool { return static_cast<cale_ctx *>(d)->cancel.load() != 0; }, h);
    return h;
}

// The CPU-only load of Phase 5, kept so older callers still work.
CALE_API cale_ctx * cale_load(const char * path, int32_t n_threads, int32_t n_ctx, int32_t repack) {
    return cale_load_dev(path, n_threads, n_ctx, repack, -1, 0, 0, 0);
}

// Which device and how many layers this load used, as text for status and replies. Returns the byte count.
CALE_API int32_t cale_placement(cale_ctx * h, char * out, int32_t cap) {
    if ((int32_t) h->placement.size() + 1 > cap) return -1;
    memcpy(out, h->placement.c_str(), h->placement.size() + 1);
    return (int32_t) h->placement.size();
}

// Stop the work in progress (called from another thread). The running call returns an error, and cale_error() says "cancelled".
CALE_API void cale_cancel(cale_ctx * h) { if (h) h->cancel.store(1); }

CALE_API void cale_free(cale_ctx * h) {
    if (!h) return;
    if (h->smpl) llama_sampler_free(h->smpl);
    for (auto & sl : h->slots) if (sl.smpl) llama_sampler_free(sl.smpl);
    llama_free(h->ctx);
    llama_model_free(h->model);
    if (h->gguf) gguf_free(h->gguf);
    delete h;
}

CALE_API int32_t cale_repacked(cale_ctx * h) { return h->repacked ? 1 : 0; }
CALE_API int32_t cale_split(cale_ctx * h) { return h->split ? 1 : 0; }

// Give back every page of the model file that is mapped into this process. Safe at any time: a weight that is still
// mapped from the file (not repacked) reads its pages again when it is used. After a repacked load this frees the
// file pages that held the originals, so the model costs its repacked size once, not twice. Returns bytes given back.
CALE_API uint64_t cale_drop_file_mapping(cale_ctx * h) {
    for (const char * name : {"per_layer_token_embd.weight", "token_embd.weight"}) {
        const ggml_tensor * t = llama_model_get_tensor(h->model, name);
        if (!t || !t->data || !t->buffer || !strstr(ggml_backend_buffer_name(t->buffer), "Mapped")) continue;
        const uintptr_t at = (uintptr_t) t->data;
#if defined(_WIN32)
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery((LPCVOID) at, &mbi, sizeof(mbi))) return 0;
        uint64_t total = 0;
        for (uintptr_t p = (uintptr_t) mbi.AllocationBase; VirtualQuery((LPCVOID) p, &mbi, sizeof(mbi)) && (uintptr_t) mbi.AllocationBase == (uintptr_t) mbi.AllocationBase; p += mbi.RegionSize) {
            if (p != (uintptr_t) mbi.BaseAddress) break;
            if (mbi.State == MEM_COMMIT && mbi.Type == MEM_MAPPED) { VirtualUnlock(mbi.BaseAddress, mbi.RegionSize); total += mbi.RegionSize; }
        }
        return total;
#elif defined(__linux__)
        FILE * f = fopen("/proc/self/maps", "r");
        if (!f) return 0;
        char line[1024];
        uint64_t freed = 0;
        while (fgets(line, sizeof(line), f)) {
            unsigned long long lo = 0, hi = 0;
            if (sscanf(line, "%llx-%llx", &lo, &hi) == 2 && at >= lo && at < hi) {
                madvise((void *) lo, hi - lo, MADV_DONTNEED);
                freed = hi - lo;
                break;
            }
        }
        fclose(f);
        return freed;
#else
        return 0;
#endif
    }
    return 0;
}

// Read a weight from the file again into its place in memory. Only repacked weights need this: they live in ordinary
// memory, so dropped pages are empty. A weight that is mapped from the file comes back by itself when it is used.
// Returns 0 if it was read, 1 if it needs nothing (mapped), -1 on error.
CALE_API int32_t cale_restore(cale_ctx * h, const char * name) {
    const ggml_tensor * t = llama_model_get_tensor(h->model, name);
    const int64_t idx = gguf_find_tensor(h->gguf, name);
    if (!t || idx < 0) { g_err = std::string("no such weight: ") + name; return -1; }
    if (!t->buffer || !strstr(ggml_backend_buffer_name(t->buffer), "REPACK")) return 1;
    const size_t size = gguf_get_tensor_size(h->gguf, idx);
    const size_t off = gguf_get_data_offset(h->gguf) + gguf_get_tensor_offset(h->gguf, idx);
    FILE * f = fopen(h->path.c_str(), "rb");
    if (!f) { g_err = "cannot open the model file"; return -1; }
    std::vector<uint8_t> buf(size);
#if defined(_WIN32)
    const bool seek_ok = _fseeki64(f, (__int64) off, SEEK_SET) == 0;      // `long` is 32 bits on Windows: the file is over 2 GB
#else
    const bool seek_ok = fseeko(f, (off_t) off, SEEK_SET) == 0;
#endif
    const bool ok = seek_ok && fread(buf.data(), 1, size, f) == size;
    fclose(f);
    if (!ok) { g_err = std::string("cannot read ") + name; return -1; }
    ggml_backend_tensor_set(const_cast<ggml_tensor *>(t), buf.data(), 0, size);
    return 0;
}


// How fast can this computer read memory, in bytes per second, with n threads? Takes about 0.15 s, needs no model.
// Answer speed is memory-bound (every active weight is read once per token), so this number predicts it.
CALE_API double cale_membw(int32_t n_threads, int32_t mb_per_thread) {
    if (n_threads < 1) n_threads = 1;
    if (mb_per_thread < 8) mb_per_thread = 8;
    const size_t n = (size_t) mb_per_thread * 1024 * 1024 / sizeof(uint64_t);
    std::vector<std::vector<uint64_t>> bufs(n_threads, std::vector<uint64_t>(n, 1));
    std::vector<uint64_t> sums(n_threads, 0);
    auto pass = [&](int t) {
        const uint64_t * p = bufs[t].data();
        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (size_t i = 0; i + 4 <= n; i += 4) { a0 += p[i]; a1 += p[i + 1]; a2 += p[i + 2]; a3 += p[i + 3]; }
        sums[t] += a0 + a1 + a2 + a3;
    };
    auto run = [&](int passes) {
        std::vector<std::thread> th;
        for (int t = 0; t < n_threads; ++t) th.emplace_back([&, t] { for (int k = 0; k < passes; ++k) pass(t); });
        for (auto & x : th) x.join();
    };
    run(1);                                                       // touch every page once
    const auto t0 = std::chrono::steady_clock::now();
    run(2);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    uint64_t total = 0;
    for (auto v : sums) total += v;
    if (total == 0) return 0.0;                                   // keeps the compiler from removing the loops
    return (double) n_threads * 2.0 * (double) n * sizeof(uint64_t) / std::max(s, 1e-9);
}

CALE_API int32_t cale_n_layers(cale_ctx * h) { return llama_model_n_layer(h->model); }
CALE_API int32_t cale_n_ctx(cale_ctx * h) { return h->n_ctx; }
CALE_API void    cale_set_threads(cale_ctx * h, int32_t n) { llama_set_n_threads(h->ctx, n, n); }

// widths: one entry per layer. 0 = skip the layer, N = first N FFN channels, <0 = full. n = 0 = full model.
CALE_API int32_t cale_set_widths(cale_ctx * h, const int32_t * widths, int32_t n) {
#ifdef CALE_STOCK
    (void) h; (void) widths; (void) n;                      // comparison build against unpatched llama.cpp
    g_err = "this build has no matryoshka patch";
    return -2;
#else
    const int32_t r = llama_set_layer_widths(h->ctx, widths, n);
    if (r != 0) g_err = "layer widths do not fit this model";
    return r;
#endif
}

// The chat as prompt text. Returns the byte count, or a negative number on error.
CALE_API int32_t cale_chat_text(cale_ctx * h, const char ** roles, const char ** contents, int32_t n,
                                char * out, int32_t cap) {
    std::vector<llama_chat_message> msgs(n);
    for (int32_t i = 0; i < n; ++i) msgs[i] = {roles[i], contents[i]};
    const char * tmpl = llama_model_chat_template(h->model, nullptr);
    const int32_t r = llama_chat_apply_template(tmpl, msgs.data(), (size_t) n, true, out, cap);
    if (r < 0) g_err = "chat template not supported";
    return r;
}

CALE_API int32_t cale_tokenize(cale_ctx * h, const char * text, int32_t add_special, int32_t * out, int32_t cap) {
    return llama_tokenize(h->vocab, text, (int32_t) strlen(text), out, cap, add_special != 0, true);
}

CALE_API int32_t cale_piece(cale_ctx * h, int32_t tok, char * out, int32_t cap) {
    return llama_token_to_piece(h->vocab, tok, out, cap, 0, false);
}

CALE_API int32_t cale_is_eog(cale_ctx * h, int32_t tok) { return llama_vocab_is_eog(h->vocab, tok) ? 1 : 0; }

// Start an answer: clear the memory, read the prompt. Call cale_next() to get tokens one by one.
// reuse: 1 = keep the start of the memory that matches the new prompt and read only the rest (a chat that grows by a turn).
// The caller says 1 only if the memory was made at the same size. cale_reused() tells how many tokens were kept.
CALE_API int32_t cale_begin3(cale_ctx * h, const int32_t * tokens, int32_t n, float temperature, float top_p, uint32_t seed, int32_t reuse) {
    if (n <= 0 || n + 1 > h->n_ctx) { g_err = "prompt does not fit the context"; return -1; }
    h->cancel.store(0);
    size_t keep = 0;
    if (reuse) {
        const size_t lim = std::min(h->kv_tokens.size(), (size_t) n - 1);        // at least one token is read, to get the first answer's logits
        while (keep < lim && h->kv_tokens[keep] == tokens[keep]) ++keep;
    }
    llama_memory_t mem = llama_get_memory(h->ctx);
    if (keep == 0 || !llama_memory_seq_rm(mem, 0, (llama_pos) keep, -1)) {       // false: the memory cannot be cut back that far
        llama_memory_clear(mem, true);
        keep = 0;
    }
    h->kv_tokens.resize(keep);
    h->reused = (int32_t) keep;
    h->n_past = (int32_t) keep;
    h->pending = -1;
    if (h->smpl) llama_sampler_free(h->smpl);
    h->smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(h->smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(h->smpl, llama_sampler_init_temp(temperature));
        if (top_p < 1.0f) llama_sampler_chain_add(h->smpl, llama_sampler_init_top_p(top_p > 1e-6f ? top_p : 1e-6f, 1));
        llama_sampler_chain_add(h->smpl, llama_sampler_init_dist(seed));
    }
    std::vector<llama_token> t(tokens + keep, tokens + n);
    return decode_tokens(h, t.data(), (int32_t) t.size()) ? 0 : -1;
}

CALE_API int32_t cale_begin2(cale_ctx * h, const int32_t * tokens, int32_t n, float temperature, float top_p, uint32_t seed) {
    return cale_begin3(h, tokens, n, temperature, top_p, seed, 0);
}

CALE_API int32_t cale_reused(cale_ctx * h) { return h->reused; }

// Forget the memory (the next prompt is read from the start).
CALE_API void cale_forget(cale_ctx * h) {
    llama_memory_clear(llama_get_memory(h->ctx), true);
    h->kv_tokens.clear();
    h->n_past = 0;
    h->reused = 0;
}

// The same without top_p (kept so older callers still work).
CALE_API int32_t cale_begin(cale_ctx * h, const int32_t * tokens, int32_t n, float temperature, uint32_t seed) {
    return cale_begin2(h, tokens, n, temperature, 1.0f, seed);
}

// The next token, or -1 on error, -2 if the context is full.
CALE_API int32_t cale_next(cale_ctx * h) {
    if (h->pending >= 0) {
        if (h->n_past + 1 > h->n_ctx) return -2;
        llama_token p = h->pending;
        if (!decode_tokens(h, &p, 1)) return -1;
    }
    const llama_token tok = llama_sampler_sample(h->smpl, h->ctx, -1);
    h->pending = tok;
    return tok;
}

// ---- answers that run together -----------------------------------------------------------------------------------
// Decoding one token reads every weight once, whatever the number of answers in the step, so a step with several answers costs little
// more than one. A *slot* is one answer: open it with its prompt, call cale_batch_step again and again, close it when it is done.
// All slots run at the size set with cale_set_widths. The prompt of a slot is read in pieces, one piece per step, next to the others' tokens.
CALE_API int32_t cale_slots(cale_ctx * h) { return (int32_t) h->slots.size(); }

static llama_sampler * make_sampler(float temperature, float top_p, uint32_t seed) {
    llama_sampler * sm = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(sm, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sm, llama_sampler_init_temp(temperature));
        if (top_p < 1.0f) llama_sampler_chain_add(sm, llama_sampler_init_top_p(top_p > 1e-6f ? top_p : 1e-6f, 1));
        llama_sampler_chain_add(sm, llama_sampler_init_dist(seed));
    }
    return sm;
}

// Start an answer in a free slot. reuse: keep the start of this slot's memory that matches the prompt (the caller says 1 only for the
// same app and the same size). Returns how many prompt tokens were kept, or -1 (g_err says why).
CALE_API int32_t cale_slot_open(cale_ctx * h, int32_t slot, const int32_t * tokens, int32_t n, float temperature, float top_p, uint32_t seed,
                                int32_t guided, int32_t reuse) {
    if (slot < 0 || slot >= (int32_t) h->slots.size()) { g_err = "no such slot"; return -1; }
    cale_slot & sl = h->slots[slot];
    if (sl.state != 0) { g_err = "slot is busy"; return -1; }
    if (n <= 0 || n + 1 > h->n_ctx) { g_err = "prompt does not fit the context"; return -1; }
    size_t keep = 0;
    if (reuse) {
        const size_t lim = std::min(sl.kv.size(), (size_t) n - 1);
        while (keep < lim && sl.kv[keep] == tokens[keep]) ++keep;
    }
    llama_memory_t mem = llama_get_memory(h->ctx);
    if (keep == 0 || !llama_memory_seq_rm(mem, slot, (llama_pos) keep, -1)) {
        llama_memory_seq_rm(mem, slot, 0, -1);
        keep = 0;
    }
    sl.kv.resize(keep);
    sl.prompt.assign(tokens, tokens + n);
    sl.read = keep;
    sl.reused = (int32_t) keep;
    sl.pending = -1;
    sl.last_idx = -1;
    sl.guided = guided != 0;
    if (sl.smpl) llama_sampler_free(sl.smpl);
    sl.smpl = make_sampler(temperature, top_p, seed);
    sl.state = 1;
    return (int32_t) keep;
}

// How many tokens of `tokens` this slot's memory already holds from the start (nothing is changed). Used to pick the slot to reuse.
CALE_API int32_t cale_slot_common(cale_ctx * h, int32_t slot, const int32_t * tokens, int32_t n) {
    if (slot < 0 || slot >= (int32_t) h->slots.size()) return 0;
    const cale_slot & sl = h->slots[slot];
    const size_t lim = std::min(sl.kv.size(), (size_t) std::max(n - 1, 0));
    size_t k = 0;
    while (k < lim && sl.kv[k] == tokens[k]) ++k;
    return (int32_t) k;
}

// The answer is done (or stopped): the slot can be used again. Its memory stays, so the same app's next turn may reuse it.
CALE_API void cale_slot_close(cale_ctx * h, int32_t slot) {
    if (slot < 0 || slot >= (int32_t) h->slots.size()) return;
    h->slots[slot].state = 0;
    h->slots[slot].pending = -1;
    h->slots[slot].last_idx = -1;
}

// Forget the memory of a slot (the next prompt in it is read from the start).
CALE_API void cale_slot_forget(cale_ctx * h, int32_t slot) {
    if (slot < 0 || slot >= (int32_t) h->slots.size()) return;
    llama_memory_seq_rm(llama_get_memory(h->ctx), slot, 0, -1);
    h->slots[slot].kv.clear();
}

// One step for every running slot. Slots with a chosen token read it. One slot that is still reading its prompt reads up to
// `prefill_budget` more of it. Then the next token of each slot is sampled, or (for a guided slot) left for the caller.
// Fills out_slot/out_kind/out_tok (up to `cap` entries) and returns how many. kind 1: out_tok is the sampled token.
// kind 2: the caller chooses: cale_slot_topk, then cale_slot_accept. kind 3: this slot failed. Returns -1 if the whole step failed.
CALE_API int32_t cale_batch_step(cale_ctx * h, int32_t prefill_budget, int32_t * out_slot, int32_t * out_kind, int32_t * out_tok, int32_t cap) {
    const int32_t n_slots = (int32_t) h->slots.size();
    const int32_t max_tokens = h->n_batch;
    llama_batch b = llama_batch_init(max_tokens, 0, 1);
    b.n_tokens = 0;
    std::vector<int32_t> deciding;                                 // slots whose logits this step produces
    std::vector<int32_t> batch_idx(n_slots, -1);
    auto add = [&](llama_token t, llama_pos pos, int32_t seq, bool logits) {
        const int32_t i = b.n_tokens++;
        b.token[i] = t; b.pos[i] = pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = seq; b.logits[i] = logits ? 1 : 0;
        return i;
    };
    for (int32_t s = 0; s < n_slots && b.n_tokens < max_tokens; ++s) {
        cale_slot & sl = h->slots[s];
        if (sl.state == 2) {
            batch_idx[s] = add(sl.pending, (llama_pos) sl.kv.size(), s, true);
            deciding.push_back(s);
        }
    }
    int32_t pf_slot = -1, pf_n = 0;
    for (int32_t s = 0; s < n_slots; ++s) {
        cale_slot & sl = h->slots[s];
        if (sl.state != 1) continue;
        pf_slot = s;
        const int32_t room = max_tokens - b.n_tokens;
        pf_n = (int32_t) std::min<size_t>({sl.prompt.size() - sl.read, (size_t) std::max(prefill_budget, 1), (size_t) std::max(room, 0)});
        for (int32_t i = 0; i < pf_n; ++i) {
            const bool last = sl.read + i + 1 == sl.prompt.size();
            const int32_t at = add(sl.prompt[sl.read + i], (llama_pos) (sl.read + i), s, last);
            if (last) batch_idx[s] = at;
        }
        if (batch_idx[s] >= 0) deciding.push_back(s);
        break;                                                      // one slot reads its prompt per step: the others keep answering
    }
    if (b.n_tokens == 0) { llama_batch_free(b); return 0; }
    const int32_t rc = llama_decode(h->ctx, b);
    if (rc != 0) {
        g_err = h->cancel.load() != 0 ? "cancelled" : "llama_decode failed";
        llama_batch_free(b);
        return -1;
    }
    for (int32_t s = 0; s < n_slots; ++s) {                        // the tokens that were read are in the memory now
        cale_slot & sl = h->slots[s];
        if (sl.state == 2) sl.kv.push_back(sl.pending);
    }
    if (pf_slot >= 0) {
        cale_slot & sl = h->slots[pf_slot];
        sl.kv.insert(sl.kv.end(), sl.prompt.begin() + sl.read, sl.prompt.begin() + sl.read + pf_n);
        sl.read += pf_n;
    }
    int32_t out = 0;
    for (int32_t s : deciding) {
        cale_slot & sl = h->slots[s];
        sl.last_idx = batch_idx[s];
        if (out >= cap) break;
        out_slot[out] = s;
        if (sl.guided) {
            sl.state = 3;
            out_kind[out] = 2; out_tok[out] = -1;
        } else {
            const llama_token t = llama_sampler_sample(sl.smpl, h->ctx, sl.last_idx);
            sl.pending = t; sl.state = 2;
            out_kind[out] = 1; out_tok[out] = t;
        }
        ++out;
    }
    llama_batch_free(b);
    return out;
}

// For a guided slot (kind 2): the k most likely next tokens with their log-probabilities. Valid until the next cale_batch_step.
CALE_API int32_t cale_slot_topk(cale_ctx * h, int32_t slot, int32_t k, int32_t * ids, float * logprob) {
    if (slot < 0 || slot >= (int32_t) h->slots.size() || h->slots[slot].last_idx < 0) { g_err = "no logits for this slot"; return -1; }
    const float * lg = llama_get_logits_ith(h->ctx, h->slots[slot].last_idx);
    if (!lg) { g_err = "no logits for this slot"; return -1; }
    const int32_t n = llama_vocab_n_tokens(h->vocab);
    k = std::min(k, n);
    std::vector<int32_t> idx(n);
    for (int32_t i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t c) { return lg[a] > lg[c]; });
    const float mx = lg[idx[0]];
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) sum += std::exp((double) (lg[i] - mx));
    const float lse = (float) (std::log(sum) + mx);
    for (int32_t i = 0; i < k; ++i) { ids[i] = idx[i]; logprob[i] = lg[idx[i]] - lse; }
    return k;
}

// The caller chose this token for a guided slot: it is read at the next step.
CALE_API int32_t cale_slot_accept(cale_ctx * h, int32_t slot, int32_t tok) {
    if (slot < 0 || slot >= (int32_t) h->slots.size()) return -1;
    cale_slot & sl = h->slots[slot];
    if ((int32_t) sl.kv.size() + 1 > h->n_ctx) { g_err = "context is full"; return -2; }
    sl.pending = tok;
    sl.state = 2;
    return 0;
}

CALE_API int32_t cale_slot_reused(cale_ctx * h, int32_t slot) { return h->slots[slot].reused; }
CALE_API int32_t cale_slot_tokens(cale_ctx * h, int32_t slot) { return (int32_t) h->slots[slot].kv.size(); }

// The k most likely next tokens, best first, for guided decoding: the caller picks one that fits its grammar, then calls cale_accept.
// ids[i] and logprob[i] get the token and its log-probability (over the whole vocabulary). Returns how many were written, or -1.
CALE_API int32_t cale_topk(cale_ctx * h, int32_t k, int32_t * ids, float * logprob) {
    const float * lg = llama_get_logits_ith(h->ctx, -1);
    if (!lg) { g_err = "no logits yet: call cale_begin first"; return -1; }
    const int32_t n = llama_vocab_n_tokens(h->vocab);
    k = std::min(k, n);
    std::vector<int32_t> idx(n);
    for (int32_t i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
    float mx = lg[idx[0]];
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) sum += std::exp((double) (lg[i] - mx));
    const float lse = (float) (std::log(sum) + mx);
    for (int32_t i = 0; i < k; ++i) { ids[i] = idx[i]; logprob[i] = lg[idx[i]] - lse; }
    return k;
}

// Take a token the caller chose: read it into the context, so the next cale_topk is for the position after it. 0 or -1.
CALE_API int32_t cale_accept(cale_ctx * h, int32_t tok) {
    if (h->n_past + 1 > h->n_ctx) { g_err = "context is full"; return -2; }
    h->pending = -1;
    llama_token t = tok;
    return decode_tokens(h, &t, 1) ? 0 : -1;
}

// Mean negative log-likelihood (nats) of tokens[first..n-1] given the text before them. Used to compare sizes.
CALE_API double cale_nll(cale_ctx * h, const int32_t * tokens, int32_t n, int32_t first) {
    if (first < 1) first = 1;
    if (n < 2) return -1.0;
    llama_memory_clear(llama_get_memory(h->ctx), true);
    h->kv_tokens.clear();
    h->n_past = 0;
    const int32_t n_vocab = llama_vocab_n_tokens(h->vocab);
    double total = 0.0;
    int32_t counted = 0;
    llama_batch b = llama_batch_init(h->n_batch, 0, 1);
    for (int32_t start = 0; start < n - 1; start += h->n_batch) {
        const int32_t k = std::min(h->n_batch, n - 1 - start);
        b.n_tokens = k;
        for (int32_t i = 0; i < k; ++i) {
            b.token[i] = tokens[start + i];
            b.pos[i] = start + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        if (llama_decode(h->ctx, b) != 0) { llama_batch_free(b); g_err = "llama_decode failed"; return -1.0; }
        for (int32_t i = 0; i < k; ++i) {
            const float * lg = llama_get_logits_ith(h->ctx, i);
            float mx = lg[0];
            for (int32_t v = 1; v < n_vocab; ++v) mx = lg[v] > mx ? lg[v] : mx;
            double sum = 0.0;
            for (int32_t v = 0; v < n_vocab; ++v) sum += std::exp((double) (lg[v] - mx));
            if (start + i + 1 >= first) {
                total += (std::log(sum) + mx) - lg[tokens[start + i + 1]];
                ++counted;
            }
        }
    }
    llama_batch_free(b);
    return counted ? total / counted : -1.0;
}

// ---- where the weights live, and keeping them in or out of RAM ------------------------------------------------
// The model file is memory-mapped, so a weight is in RAM only while its pages are. These calls are the only
// place that talks to the OS about pages. Linux and macOS use madvise and mincore. Windows uses the working set.

namespace {
uint64_t page_size() {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return (uint64_t) sysconf(_SC_PAGESIZE);
#endif
}
}  // namespace

// Where a named weight sits in memory. 0 on success.
CALE_API int32_t cale_tensor_range(cale_ctx * h, const char * name, uint64_t * addr, uint64_t * nbytes) {
    const ggml_tensor * t = llama_model_get_tensor(h->model, name);
    if (!t || !t->data) return -1;
    // A weight on a GPU has no address in this process: report its size and address 0 (the page calls ignore address 0).
    const char * buf = t->buffer ? ggml_backend_buffer_name(t->buffer) : "";
    const bool in_ram = strstr(buf, "CPU") || strstr(buf, "Mapped") || strstr(buf, "REPACK");
    *addr = in_ram ? (uint64_t) (uintptr_t) t->data : 0;
    *nbytes = (uint64_t) ggml_nbytes(t);
    return 0;
}

// op 0: let the OS take these pages out of RAM (they read back from the file when used again). Returns bytes.
// op 1: read these pages into RAM now. Returns bytes.
// op 2: how many bytes of this range this process has in RAM right now.
CALE_API uint64_t cale_pages(int32_t op, uint64_t addr, uint64_t len) {
    const uint64_t ps = page_size();
    const uint64_t lo = (addr + ps - 1) / ps * ps;            // inside the range: only whole pages
    const uint64_t hi = (addr + len) / ps * ps;
    if (op == 1) {
        volatile unsigned char sink = 0;
        const uint64_t from = addr / ps * ps, to = addr + len;
#if defined(_WIN32)
        WIN32_MEMORY_RANGE_ENTRY e = {(PVOID) from, (SIZE_T) (to - from)};
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &e, 0);
#else
        madvise((void *) from, to - from, MADV_WILLNEED);
#endif
        for (uint64_t a = from; a < to; a += ps) sink = sink + *(volatile unsigned char *) a;
        (void) sink;
        return len;
    }
    if (hi <= lo) return 0;
    if (op == 0) {
#if defined(_WIN32)
        VirtualUnlock((LPVOID) lo, (SIZE_T) (hi - lo));       // on pages that are not locked this removes them from the working set
#else
        madvise((void *) lo, hi - lo, MADV_DONTNEED);
#endif
        return hi - lo;
    }
    uint64_t resident = 0;
#if defined(_WIN32)
    const uint64_t n = (hi - lo) / ps, chunk = 4096;
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> v(chunk);
    for (uint64_t i = 0; i < n; i += chunk) {
        const uint64_t k = std::min(chunk, n - i);
        for (uint64_t j = 0; j < k; ++j) v[j].VirtualAddress = (PVOID) (lo + (i + j) * ps);
        if (!QueryWorkingSetEx(GetCurrentProcess(), v.data(), (DWORD) (k * sizeof(v[0])))) return 0;
        for (uint64_t j = 0; j < k; ++j) resident += v[j].VirtualAttributes.Valid ? ps : 0;
    }
#elif defined(__linux__)
    // Pages this process has mapped right now (not the shared page cache, which is what mincore would tell us).
    const uint64_t n = (hi - lo) / ps;
    FILE * f = fopen("/proc/self/pagemap", "rb");
    if (f) {
        std::vector<uint64_t> ent(4096);
        for (uint64_t i = 0; i < n; i += ent.size()) {
            const uint64_t k = std::min<uint64_t>(ent.size(), n - i);
            if (fseeko(f, (off_t) ((lo / ps + i) * 8), SEEK_SET) != 0 || fread(ent.data(), 8, k, f) != k) break;
            for (uint64_t j = 0; j < k; ++j) resident += (ent[j] >> 63) ? ps : 0;
        }
        fclose(f);
        return resident;
    }
    std::vector<unsigned char> vec(n);
    if (mincore((void *) lo, hi - lo, vec.data()) != 0) return 0;
    for (uint64_t i = 0; i < n; ++i) resident += (vec[i] & 1) ? ps : 0;
#else
    const uint64_t n = (hi - lo) / ps;
    std::vector<unsigned char> vec(n);
    if (mincore((void *) lo, hi - lo, (char *) vec.data()) != 0) return 0;
    for (uint64_t i = 0; i < n; ++i) resident += (vec[i] & 1) ? ps : 0;
#endif
    return resident;
}
