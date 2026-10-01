"""Greedy generation and letter-choice scoring that pass image token types to the decoder.

Works for any loaded level of an elastic model: skipped layers simply never
write to their KV cache, which is consistent within one generation.
"""
import mlx.core as mx

from .io import encode

EOS = {1, 106}


def _prefill(model, ids, kw, cache):
    emb = model.get_input_embeddings(input_ids=ids, **kw)
    return model.language_model.model(
        None, inputs_embeds=emb.inputs_embeds, cache=cache, per_layer_inputs=emb.per_layer_inputs,
        mm_token_type_ids=kw.get("mm_token_type_ids")
    )


def generate(model, processor, rec, max_tokens=256):
    ids, kw = encode(processor, rec, with_answer=False, max_tokens=4096)
    lm = model.language_model
    cache = lm.make_cache()
    h = _prefill(model, ids, kw, cache)
    tok = mx.argmax(lm.logits_from_hidden(h[:, -1:]), axis=-1)
    mx.eval(tok)
    out = []
    for _ in range(max_tokens):
        t = int(tok.item())
        if t in EOS:
            break
        out.append(t)
        h = lm.model(tok.reshape(1, 1), cache=cache)
        tok = mx.argmax(lm.logits_from_hidden(h[:, -1:]), axis=-1)
        mx.async_eval(tok)
    return processor.tokenizer.decode(out, skip_special_tokens=True)


def choose_letter(model, processor, rec, letter_ids):
    """Pick the answer letter with the highest next-token log-prob."""
    ids, kw = encode(processor, rec, with_answer=False, max_tokens=4096)
    h = _prefill(model, ids, kw, None)
    lg = model.language_model.logits_from_hidden(h[:, -1])[0]
    scores = lg[mx.array([letter_ids[c] for c in rec["choices"]])]
    return rec["choices"][int(mx.argmax(scores).item())]
