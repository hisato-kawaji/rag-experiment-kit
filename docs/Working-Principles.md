# Working Principles (作業方針)

このリポジトリで作業する際のルール。守ると Claude Code (or 別の作業者) が継続作業しやすい。

## 1. 1 論文 1 ブランチ 1 issue 1 PR

- 新しい論文を取り込む = ブランチ `paper/<name>`
- 新しいインフラ機能 = ブランチ `feat/<short>` or `infra/<short>`
- 1 PR は 1 issue を closes する
- レビューしやすさ優先で、論文実装途中の WIP は draft PR にする

## 2. Pipeline 系の変更は必ず eval を回してから merge

```bash
task synth-eval -- --n 20    # 既に生成済みなら skip OK
task run -- --pipeline <new>
task compare                 # baseline と並べる
```

PR description に compare 結果の表を貼る。Ragas スコアが既存 baseline 比で **明確に劣化していたら merge しない**(理由を明記する場合のみ例外)。

## 3. Protocol を破る変更は議論してから

`LLM`, `Embedding`, `VectorStore`, `Pipeline`, `Source`, `Retriever`, `Reranker`, `Parser` の interface 変更は **全 backend / 全 pipeline に影響する**。

破る場合:
1. 先に issue で背景と影響範囲を書く
2. 既存の実装をすべて新 Protocol に追従させる
3. CHANGELOG (Wiki) に "BREAKING" として記録

非破壊で広げる(メソッド追加 + デフォルト実装提供)なら issue 不要。

## 4. テクニック実装は `techniques/<name>/` に閉じる

- ✅ `src/rag/techniques/<name>/{prompts,extractor,retriever,pipeline,...}.py`
- ✅ `papers/<name>_<id>/{paper.pdf, meta.yaml, NOTES.md}`
- ✅ `configs/pipelines/<name>.yaml`
- ❌ 既存 pipeline (baseline.py / graphrag/) を編集しない (上書きしたい場合は新 pipeline を作る)
- ❌ `pipelines/__init__.py` の factory 以外の共有層に手を入れない (再利用したいヘルパは `retrievers/` か `rerankers/` に)

## 5. Commit メッセージは Conventional Commits

```
<type>(<scope>): <subject>

<optional body>
<optional footer with `Closes #N`>
```

| type | 用途 |
|---|---|
| `feat` | 新機能 (新 pipeline / 新 source / 新 adapter / etc.) |
| `fix` | バグ修正 |
| `refactor` | 振る舞い変えない構造変更 |
| `perf` | パフォーマンス改善 |
| `docs` | README / Wiki / docstring |
| `test` | テスト追加・修正 |
| `chore` | 依存更新、設定変更 |

例:
```
feat(techniques): add HippoRAG self-built implementation (Closes #9)
fix(qdrant): use query_points API (search() removed in client>=1.16)
refactor(retriever): extract dense logic from BaselineRAG (prep for #12)
```

## 6. Lazy import を死守する

CLI / pipeline factory で、重い dependency をモジュールトップで import しない:

```python
# ❌ 悪い
from .techniques.graphrag.build import build_graphrag_index

@app.command("build-graph")
def cmd_build_graph(...):
    build_graphrag_index(...)
```

```python
# ✅ 良い
@app.command("build-graph")
def cmd_build_graph(...):
    from .techniques.graphrag.build import build_graphrag_index
    build_graphrag_index(...)
```

理由: baseline だけ走らせるユーザーが leidenalg / graspologic / colbert 等の重 deps を入れずに済む。`pyproject.toml` の optional extras と組み合わせて modular install を実現するため。

## 7. テストは "落ちる単体テスト" を最小限

- ロジック純な部分 (chunking, graph build, etc.) には pytest unit を書く
- LLM / vector store を呼ぶ統合テストは書かない (高コスト + flaky)
- Pipeline の動作確認は `task run --pipeline <X>` の eval 完走 + Ragas 値の妥当性で代替

## 8. 設定は .env, パイプライン固有は CLI 引数

- credential / URL / モデル名: `.env`
- pipeline 固有パラメタ (top_k, mode, etc.): CLI 引数 or `configs/pipelines/<name>.yaml`
- ❌ 設定ファイルを 3 階層以上にネストしない
- ❌ Hydra を Phase 2 までは積極使用しない (sweep が必要になったら検討)

## 9. ファイルやデータを git に入れない

`.gitignore` で:
- `.env`, `.venv/`, `data/`, `experiments/runs/`, `models/`, `*.pdf` は ignore
- 論文 PDF は `task new-paper` で再 DL できるので追跡しない
- eval 結果は run dir に保存、共有したい時のみ artifacts として PR description に貼る

## 10. Claude Code との協働

- 新しい論文取り込みは Phase 2 で `/implement-paper <name>` (Claude slash command, [issue #7](https://github.com/hisato-kawaji/rag-experiment-kit/issues/7)) で半自動化される予定
- それまでの間は: Claude に「papers/<name>_<id>/paper.pdf を読んで src/rag/techniques/<name>/ を埋めて」と頼む
- Architecture / Working-Principles ページ ([[Architecture]], このページ) を Claude のコンテキストに毎回渡すと品質が安定する

## 11. PR レビューチェックリスト

- [ ] Protocol を破ってない (or 破る合意済み)
- [ ] Lazy import 守ってる
- [ ] 該当 issue を `Closes #N` で参照
- [ ] eval 結果 (Ragas 比較表) を PR description に貼った
- [ ] 新規依存は `pyproject.toml` の適切な extras に入れた
- [ ] テストが落ちてない (`task test`)
- [ ] `task lint` 通る
