# Research references

These are the four papers cited in Threadmark's original latent-memory design notes. They informed that proposal and provide research background; the released library does not reproduce their models or reported results. The current implementation maintains typed JSON state, uses its source identifiers to select original evidence, and sends that evidence to a text-based reader. It does not train a neural compressor or inject continuous memory into a decoder.

See the [Threadmark research note](threadmark.md) for the implemented method, mathematical formulation, experiments, and separately labelled future proposal. Machine-readable citations are in [references.bib](references.bib).

## 1. Recurrent Memory Transformer

**Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. 2022.** *Recurrent Memory Transformer*. NeurIPS 2022. [Paper](https://arxiv.org/abs/2207.06881) · [Version reviewed: v2](https://arxiv.org/html/2207.06881v2).

The model carries continuous memory between sequential text segments. Read and write memory tokens surround each segment; updated write representations become the next segment's memory. Training can propagate gradients through these recurrent states.

**Relationship:** this informed the original proposal to update a bounded memory after each historical chunk. Threadmark's released state is a validated JSON record updated through model calls; it has no recurrent Transformer state or backpropagation through memory.

## 2. In-context Autoencoder

**Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2023 preprint; ICLR 2024.** *In-context Autoencoder for Context Compression in a Large Language Model*. [Paper](https://arxiv.org/abs/2307.06945) · [Version reviewed: v4](https://arxiv.org/html/2307.06945v4).

ICAE uses a LoRA-adapted LLM encoder to turn context into continuous memory slots consumed by an untouched LLM decoder. Training combines autoencoding, text continuation, and instruction tuning. A small number of added parameters still relies on the underlying LLM encoder.

**Relationship:** this informed the original learned-compression and frozen-reader proposal. The release passes selected original text to its reader; it does not implement ICAE's encoder, memory slots, or training objectives.

## 3. Prefix-Tuning

**Xiang Lisa Li and Percy Liang. 2021.** *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. [Paper](https://arxiv.org/abs/2101.00190) · [Version reviewed: v1](https://arxiv.org/html/2101.00190v1).

Prefix-tuning optimizes continuous task-specific prefix activations while freezing the language model. The paper's method optimizes prefix representations across Transformer layers, rather than only learning ordinary input-token embeddings.

**Relationship:** this supports the original idea of conditioning a frozen decoder through trainable continuous representations. The proposed Threadmark connector would instead supply history-dependent soft tokens at the input. Neither interface is implemented in the released JSON-and-evidence method.

## 4. MemGen

**Guibin Zhang, Muxin Fu, and Shuicheng Yan. 2025.** *MemGen: Weaving Generative Latent Memory for Self-Evolving Agents*. [Paper](https://arxiv.org/abs/2509.24704) · [Version reviewed: v2](https://arxiv.org/html/2509.24704v2).

MemGen uses a learned memory trigger and memory weaver alongside a frozen reasoner. The trigger chooses invocation points during reasoning; the weaver generates a latent token sequence that conditions subsequent generation. Experience is learned through the weaver's parameters.

**Relationship:** this was cited as related work for the original latent-memory proposal. Threadmark's release has no learned invocation trigger, latent weaver, or memory insertion during decoding. Its query-time evidence selection is an application-level operation before a reader call.

## Attribution and scope

The mathematical notation in the Threadmark research note describes this project's method and proposal; it is not presented as equations copied from these papers. Their performance claims belong to their respective experiments and do not establish Threadmark's accuracy, speed, novelty, or compression guarantees. The papers remain available from their authors through the primary links above; no external paper PDF is redistributed here.
