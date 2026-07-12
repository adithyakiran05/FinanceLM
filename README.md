# FinanceLM Pre-Training Pipeline

This repository contains the architecture and data pipeline for pre-training a specialized Small Language Model (SLM) from scratch, focused entirely on quantitative finance and macroeconomics.

## Architecture Pipeline

1. **Data Collection & Cleaning**: Highly parallelized multiprocessing scripts to ingest, deduplicate, and clean SEC 10-K/10-Q filings, earnings call transcripts, Federal Reserve minutes, and quantitative finance textbooks.
2. **Custom Tokenizer (`scripts/train_tokenizer.py`)**: Builds a custom Byte-Level BPE tokenizer tailored perfectly to the financial lexicon (e.g. `amortization`, `EBITDA`).
3. **Binary Serialization (`scripts/build_corpus.py`)**: Converts the massive 283-million-word raw text dataset into a hyper-efficient `numpy.memmap` 16-bit binary file (`corpus.bin`) to allow out-of-core training without crashing RAM.
4. **Pure PyTorch Architecture (`scripts/train_model.py`)**: A from-scratch, raw PyTorch implementation of the GPT architecture (~20 Million Parameters) featuring:
   - Custom `MultiHeadAttention` logic.
   - Custom Transformer `Block` with LayerNorm and Residual connections.
   - A manual training loop optimized for CUDA mixed-precision training.
