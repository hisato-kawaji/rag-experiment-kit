# 新しい論文の取り込み手順

`task new-paper` で雛形を生成 → 論文を読みながら埋める → 評価。標準的な所要時間は **1 日 / 論文**(Phase 2 の `/implement-paper` slash command 完成後は **半日 / 論文** を狙う)。

## 0. 前提

- 論文の arXiv ID を控える(例: `2405.14831`)
- 取り込み名は短い snake_case(例: `hipporag`)
- どの既存ピース (LLM / Embedding / VectorStore) を再利用し、何を自前実装するか決めておく

## 1. ブランチ + scaffold

```bash
git checkout -b paper/hipporag
task new-paper -- --id 2405.14831 --name hipporag
```

これで以下が一気に生成される:

```
papers/hipporag_2405.14831/
├── paper.pdf            # 自動 DL
├── meta.yaml            # status: planned, key_ideas: TODO
└── NOTES.md             # 実装メモのテンプレ

src/rag/techniques/hipporag/
├── __init__.py
└── pipeline.py          # HipporagPipeline スケルトン

configs/pipelines/hipporag.yaml
```

## 2. 論文を読む + 実装を計画

`papers/hipporag_2405.14831/NOTES.md` の **Reading map** セクションを埋める。論文の §X.Y がどの実装ファイルに対応するかを書く:

```markdown
## Reading map
- §3.1 Indexing → src/rag/techniques/hipporag/extractor.py + graph.py
- §3.2 Retrieval → src/rag/techniques/hipporag/retriever.py (Personalized PageRank)
- §3.3 Inference → src/rag/techniques/hipporag/pipeline.py
```

`meta.yaml` の `key_ideas` も埋める。

## 3. モジュール構成 (推奨パターン)

GraphRAG 実装 (`src/rag/techniques/graphrag/`) と同じ構成にすると比較しやすい:

```
src/rag/techniques/<name>/
├── __init__.py
├── prompts.py           # LLM プロンプト集
├── extractor.py         # 論文のインデックス段階 (もし LLM 抽出があれば)
├── graph.py             # データ構造構築 (graph / tree / etc.)
├── summarizer.py        # 中間表現 (もしあれば、例: community summary)
├── store.py             # 永続化 (pickle / JSON)
├── build.py             # build orchestrator: 上の全部を呼んで artifact 出力
├── retriever.py         # クエリ時のロジック
└── pipeline.py          # Pipeline protocol を満たす公開 class
```

全部要らない場合もある (例: Contextual Retrieval は extractor / graph / community が不要)。最小は `prompts.py` + `pipeline.py` + 必要なら `retriever.py`。

## 4. 実装

### 4a. Pipeline class

`Pipeline` Protocol (`name: str, answer(query) -> Answer`) を満たす:

```python
# src/rag/techniques/hipporag/pipeline.py
from __future__ import annotations
from ...types import Answer
from .retriever import HippoRAGRetriever

class HippoRAGPipeline:
    name = "hipporag"

    def __init__(self, *, top_k: int = 5, **_):
        self.top_k = top_k
        self._retriever = HippoRAGRetriever(...)

    def answer(self, query: str) -> Answer:
        return self._retriever.search(query, k=self.top_k)
```

### 4b. Pipelines factory への登録

`src/rag/pipelines/__init__.py` の `build_pipeline` に lazy import で追加:

```python
def build_pipeline(name: str, **kwargs):
    if name == "baseline":
        return BaselineRAG(**kwargs)
    if name == "graphrag":
        from ..techniques.graphrag.pipeline import GraphRAGPipeline
        return GraphRAGPipeline(**kwargs)
    if name == "hipporag":                        # ← 追加
        from ..techniques.hipporag.pipeline import HippoRAGPipeline
        return HippoRAGPipeline(**kwargs)
    raise ValueError(...)
```

### 4c. CLI に build / 専用コマンドが必要なら追加

GraphRAG のように indexing 段階が重い場合は `task build-X` を追加:

```python
# src/rag/cli.py
@app.command("build-hipporag")
def cmd_build_hipporag(out_dir: Path = ...):
    from .techniques.hipporag.build import build_hipporag_index
    build_hipporag_index(out_dir)
```

## 5. 評価

```bash
task synth-eval -- --n 20            # 既存と同 eval set を共有
task run -- --pipeline hipporag      # 新手法を eval set 対象に
task run -- --pipeline baseline      # 比較対象 (既に走らせていれば skip)
task compare                         # 横並び
```

期待: baseline と GraphRAG vs HippoRAG が並ぶ表が出る。

## 6. PR

PR description の必須セクション:

```markdown
## 取り込んだ論文
arXiv:2405.14831 — HippoRAG: Neurobiologically Inspired Long-Term Memory ...

## 実装の概要
- Indexing: passage + concept の 2 層 graph
- Retrieval: Personalized PageRank with query entities as seed
- 実装位置: src/rag/techniques/hipporag/

## Phase 1 simplifications
- OpenIE は GraphRAG の extractor 流用 (NER 用 spaCy は別 issue で導入予定)
- ...

## Eval (5 質問, llama3.1:8b)
| metric | baseline | graphrag/global | hipporag |
|---|---|---|---|
| answer_relevancy | 0.86 | 0.89 | 0.91 |
| context_precision | 1.00 | 1.00 | 1.00 |
| context_recall | 0.74 | 1.00 | 0.95 |
| elapsed | 43s | 836s | 280s |

## TODO (Phase 2 以降に分離)
- Multi-hop な合成 eval set で再検証 (#13 ブロック)
- entity index 共有 (#2 と統合)

Closes #9
```

## 7. wiki 更新

`Roadmap` ページの該当 issue を ✅ に変更、必要なら `Architecture` ページの "Pipelines" セクションに 1 行追加。

---

## Phase 2 で半自動化される

`/implement-paper <name>` Claude slash command ([issue #7](https://github.com/hisato-kawaji/rag-experiment-kit/issues/7)) が完成すると:

1. `task new-paper` で scaffold
2. `/implement-paper hipporag` を Claude で実行
3. Claude が PDF を読み、§ ごとに該当ファイルを埋める
4. 開発者は `meta.yaml` の `phase1_simplifications` を確認 + eval を回すだけ

ステップ 4-6 が大幅短縮される。
