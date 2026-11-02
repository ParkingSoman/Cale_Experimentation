# EleutherAI delphi: notes for raw-neuron use (2026-10-19)

Installed: `eai-delphi 0.1.3`, with `--no-deps`, in `research/clean/tools/.venv`.
Source copy: `src/delphi/`. Dry run: `pilots/t2_dry.py` (passes, no API call).

## (a) What we would call

| Step | Class or function | File in `delphi/` |
|---|---|---|
| Hook the neurons and save them | `LatentCache(model, hookpoint_to_sparse_encode, batch_size, transcode=True, filters=...)`, then `.run(n_tokens, tokens)`, `.save_splits(n_splits, save_dir)`, `.save_config(save_dir, cfg, model_name)` | `latents/cache.py` (class at line 177) |
| Read the hook | `collect_activations(model, hookpoints, transcode)` | `latents/collect_activations.py` |
| Load the cache and build examples | `LatentDataset(raw_dir, sampler_cfg, constructor_cfg, tokenizer, modules, latents)` | `latents/loader.py` (line 119) |
| Pick top examples and non-active examples | `constructor(...)`, `pool_max_activation_windows`, `random_non_activating_windows` | `latents/constructors.py` (231, 84, 644) |
| Sample train and test sets | `sampler(record, sampler_cfg, tokenizer)` | `latents/samplers.py` |
| Explain | `DefaultExplainer(client, threshold=0.3)`, called as `await explainer(record)` | `explainers/default/default.py`, base in `explainers/explainer.py` |
| Score | `DetectionScorer(client, n_examples_shown=...)`, `FuzzingScorer(client, ...)`, called as `await scorer(record)` (the record needs `.explanation`) | `scorers/classifier/detection.py`, `fuzz.py`, `classifier.py` |
| Run many records | `Pipeline(loader, Pipe(explainer), Pipe(scorer)).run(max_concurrent)` | `pipeline.py` |
| Config | `CacheConfig`, `ConstructorConfig`, `SamplerConfig` | `config.py` |

## (b) What an adapter must provide

Raw neurons need no encoder. Use the identity.

1. Hookpoint name. Use `"model.layers.28.mlp.down_proj"` with `transcode=True`.
   The hook then stores the module INPUT, which is `act_fn(gate) * up`, shape `[batch, seq, 1536]` for SmolLM2-135M.
   `LatentCache.__init__` wants `hookpoint_to_sparse_encode = {hookpoint: (lambda x: x)}`.
2. Tokens. `tokens`: int tensor `[n_seq, ctx_len]`. `ctx_len` is `cache_ctx_len` (default 256). `ConstructorConfig.example_ctx_len` (default 32) must divide it.
3. On-disk format (written by `save_splits`, read by `TensorBuffer.load`, `latents/loader.py` line 84):
   one folder per hookpoint, files named `{start}_{end}.safetensors`, plus `config.json` that holds `model_name` and the cache config.
   Each file has `locations` (uint16 or uint32, `[n, 3]` = batch, position, neuron index minus `start`), `activations` (float16, `[n]`), and `tokens` (`[n_seq, ctx_len]`).
   Only values with `abs > 1e-5` are kept (`cache.py` line 141).
4. Use `filters={hookpoint: tensor_of_neuron_ids}` to keep only the neurons we study. A dense layer stored whole is huge (every neuron fires a little at every token).
5. Window math. `constructor_cfg.min_examples` defaults to 200. A neuron with fewer windows is dropped (returns `None`). Lower it for rare neurons.

Sign problem. SwiGLU neurons are signed. Delphi treats a large POSITIVE value as "active".
`normalize_activations` in `latents/samplers.py` line 16 does `clamp(0, 10)`, so negative values become 0.
`pool_max_activation_windows` ranks by max (not abs). To explain the negative side, run a second cache with the activations multiplied by -1 (identity encoder `lambda x: -x`) and name it as a separate hookpoint or folder.

## (c) Anthropic or OpenRouter client

- OpenRouter: yes, already built in. `delphi.clients.OpenRouter(model, api_key=..., max_tokens=3000, temperature=1.0)` (`clients/openrouter.py`). It posts chat messages to the OpenRouter URL.
  Use a model id such as `anthropic/claude-...`. No logprobs (so `log_prob=True` in scorers does not work). Default retry count is 1 (`max_retries=1`). The class-level timeout is 5 s but each call passes 100 s.
- Anthropic direct: no client in the package. Write a small subclass of `delphi.clients.client.Client`:
  `async def generate(self, prompt, **kw) -> Response(text=...)`. `prompt` is a list of `{"role", "content"}` messages; the first one has role `system`, which the Anthropic API wants in its own `system=` field. Pop `temperature`, `max_tokens`, `logprobs`, `schema` from `kw`. About 20 lines. `FakeClient` in `pilots/t2_dry.py` shows the shape.
- Both keys: `OPENROUTER_API_KEY` is set on this Mac. `ANTHROPIC_API_KEY` is not set.

## (d) Blockers

| Blocker | Detail | Workaround |
|---|---|---|
| `vllm` import | `delphi/clients/__init__.py` imports `offline.py`, which imports vllm. This breaks every import of `delphi.clients`, `delphi.explainers` and `delphi.scorers`. | `pilots/t2_stubs.py` puts stub modules in `sys.modules`. No edit of the clone. Works (see `t2_dry.py`). |
| `faiss`, `sentence_transformers` | Imported at top of `latents/constructors.py`. Needed only for `non_activating_source="FAISS"`. | Stubs. Use `non_activating_source="random"`. |
| `sparsify` (an SAE library) | Imported at top of `latents/neighbours.py` and `sparse_coders/load_sparsify.py`. | Stub only. We do not install it. Never use `load_sparsify`, `neighbours` source, or `__main__.py`. |
| `tiktoken` | Needed by the simulator module. | Installed. |
| Signed activations | See (b). | Run a negated pass. |
| Reader code | `__main__.py` assumes SAE models. | Do not use the CLI. Call the classes from our own script. |

Verified with `pilots/t2_dry.py`: DefaultExplainer parsed `[EXPLANATION]:` and DetectionScorer produced predictions through a fake client. Cost: 0 dollars.
Note: the install put a copy in the venv `site-packages`. Python `-I` still imports it from there. Read the clone only as text.
