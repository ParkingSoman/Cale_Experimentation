"""Stub out heavy/rejected deps so delphi imports on a Mac without vllm/faiss/sparsify."""
import sys, types
from unittest.mock import MagicMock
for name in ["vllm","vllm.distributed","vllm.distributed.parallel_state","vllm.inputs","faiss",
             "sentence_transformers","sparsify","sparsify.data","sparsify.sparse_coder","bitsandbytes"]:
    sys.modules.setdefault(name, MagicMock())
