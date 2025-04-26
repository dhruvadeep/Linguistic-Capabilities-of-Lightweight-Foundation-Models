import pandas as pd
from datasets import load_dataset


def build_conll2003_df(split: str) -> pd.DataFrame:
    """
    Load a split of the eriktks/conll2003 dataset and return a DataFrame with:
      - text_original : sentence as one string
      - pos_tags_text : tokens with [POS] appended
      - ner_tags_text : tokens with [NER] appended
    """
    ds = load_dataset("eriktks/conll2003", split=split, trust_remote_code=True)

    pos_labels = ds.features["pos_tags"].feature.names
    ner_labels = ds.features["ner_tags"].feature.names

    rows = []
    for tokens, pos_ids, ner_ids in zip(ds["tokens"], ds["pos_tags"], ds["ner_tags"]):
        text_orig = " ".join(tokens)
        pos_txt = " ".join(f"{t}[{pos_labels[p]}]" for t, p in zip(tokens, pos_ids))
        ner_txt = " ".join(f"{t}[{ner_labels[n]}]" for t, n in zip(tokens, ner_ids))
        rows.append(
            {
                "text_original": text_orig,
                "pos_tags_text": pos_txt,
                "ner_tags_text": ner_txt,
            }
        )

    return pd.DataFrame(rows)


splits = ["train", "validation", "test"]
df = pd.concat([build_conll2003_df(s) for s in splits], ignore_index=True)

df_1 = df.copy()
df_2 = df.copy()

df_1 = df_1.rename(columns={"text_original": "question", "pos_tags_text": "answers"})
df_1 = df_1[["question", "answers"]]

df_2 = df_2.rename(columns={"text_original": "question", "ner_tags_text": "answers"})
df_2 = df_2[["question", "answers"]]

df_1.to_json("test_1.jsonl", orient="records", lines=True, force_ascii=False)
df_2.to_json("test_2.jsonl", orient="records", lines=True, force_ascii=False)
