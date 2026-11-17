# Arabic Word-Sense Embeddings

**Do transformer models understand what an Arabic word means in context?**

Many Arabic words change meaning with context. **هدف** can be a football *goal* or a company's *objective*. **شبكة** can be the *net* of a goal or an internet *network*. Static embeddings (Word2Vec, GloVe) give each word one fixed vector. Transformer models produce **contextual** embeddings, which should change with the sentence.

This project extracts contextual embeddings from four model families and measures how well each one separates the different senses of polysemous Arabic words.

## Models compared

| Model | Architecture | Why it is included |
|---|---|---|
| [AraBERT v02](https://huggingface.co/aubmindlab/bert-base-arabertv02) | BERT encoder | Arabic-specific pre-training and vocabulary |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | XLM-RoBERTa encoder | Strong multilingual retrieval model with an 8K context window |
| [multilingual-E5-base](https://huggingface.co/intfloat/multilingual-e5-base) | XLM-RoBERTa encoder | Popular multilingual embedding model (`passage: ` prefix) |
| [Qwen2.5-1.5B / 7B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | Causal decoder (LLM) | Shows how a general-purpose LLM represents Arabic |

## What the project does

1. **Tokenization analysis**: how many tokens each model needs for the same Arabic sentence, and what those tokens look like (whole words vs. byte-level fragments).
2. **Embedding extraction**: token vectors from the last hidden layer. Words split into several sub-word tokens are averaged into a single word vector using the tokenizer's character offsets.
3. **Word-sense test**: for five polysemous words (هدف، شبكة، عين، ذهب، قلب) there are three sentences each: an anchor, one with the **same** sense, and one with a **different** sense. For every model we compute:
   - `sim_same_sense`: cosine(anchor, same-sense)
   - `sim_different_sense`: cosine(anchor, different-sense)
   - `sense_separation = sim_same_sense - sim_different_sense` (higher is better)
   - `accuracy`: the share of words where same-sense uses are closer than different-sense uses
4. **Frozen vs. trainable embeddings**: freezing the input embedding layer and the effect on the number of trainable parameters.
5. **Maximum sequence length**: what happens when the input goes past 512 tokens, and how truncation fixes it.

Separation is used instead of raw cosine similarity because every model has its own embedding space. Some models (E5, for example) give high similarity to almost everything, so raw values can't be compared across models.

## Results

Running the notebook saves these files to `results/`:

- `sense_separation.png`: per-word and average sense separation for each model
- `model_summary.csv`: average similarities, separation and accuracy per model
- `word_sense_results.csv`: full per-word results
- `tokenization.csv`: token counts per model

![Sense separation by model](results/sense_separation.png)

## Project structure

```
arabic-word-sense-embeddings/
├── arabic_contextual_embeddings.ipynb   # main analysis notebook
├── src/
│   └── contextual_embeddings.py         # model loading, word/sentence embeddings, evaluation
├── data/
│   └── polysemous_words.json            # word-sense test set (5 words x 3 sentences)
├── results/                             # generated tables and figures
├── requirements.txt
└── LICENSE
```

## How to run

```bash
git clone https://github.com/muradzughair/arabic-word-sense-embeddings.git
cd arabic-word-sense-embeddings
pip install -r requirements.txt
jupyter notebook arabic_contextual_embeddings.ipynb
```

- AraBERT, BGE-M3, E5 and Qwen2.5-1.5B run on a normal laptop CPU. The first run downloads the models, a few GB in total.
- Qwen2.5-7B needs a GPU with about 16 GB of memory (Google Colab T4 works in bfloat16). Set `RUN_LARGE_MODELS = True` in the setup cell to include it.

## Adding your own test words

Add an entry to `data/polysemous_words.json`:

```json
{
  "word": "عين",
  "sense_a": "eye",
  "sense_b": "water spring",
  "anchor": "فحص الطبيب عين المريض بعد العملية",
  "same_sense": "أصيبت عين اللاعب بكدمة خلال المباراة",
  "different_sense": "شرب المسافرون من عين ماء باردة في الجبل"
}
```

The target word must appear as a separate word, not attached to a prefix like و or ال, in all three sentences.

## Future work

- A larger word-sense dataset with more words and more sentences per sense
- Comparing hidden layers, not only the last one
- Visualizing word vectors with PCA or t-SNE
- Static embedding baselines (AraVec, fastText) for comparison

## Tech stack

Python, PyTorch, Hugging Face Transformers, pandas, matplotlib

## License

MIT
