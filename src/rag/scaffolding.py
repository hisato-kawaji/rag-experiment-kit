"""Scaffolding for `task new-paper` — generates paper dir + technique skeleton + config."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.request import urlretrieve

from .logging import log

PAPER_META_TEMPLATE = """\
arxiv_id: "{arxiv_id}"
title: "TODO — paste title"
authors: []
year: TODO
url: https://arxiv.org/abs/{arxiv_id}
pdf: paper.pdf

status: planned
implementation_module: rag.techniques.{name}
implementation_path: src/rag/techniques/{name}/
config: configs/pipelines/{name}.yaml

key_ideas:
  - TODO

phase1_simplifications: []
phase2_followups: []
"""

PAPER_NOTES_TEMPLATE = """\
# {name} — implementation notes

PDF: [./paper.pdf](./paper.pdf)
Module: `src/rag/techniques/{name}/`

## Pipeline summary
TODO — fill in after reading the paper.

## Reading map
TODO
"""

TECHNIQUE_INIT_TEMPLATE = """\
from .pipeline import {cls_name}Pipeline

__all__ = ["{cls_name}Pipeline"]
"""

TECHNIQUE_PIPELINE_TEMPLATE = '''\
"""TODO: implement {name} from arXiv:{arxiv_id}.

Skeleton conforms to rag.pipelines.base.Pipeline protocol.
"""

from __future__ import annotations

from typing import Any

from ...types import Answer


class {cls_name}Pipeline:
    name = "{name}"

    def __init__(self, *, top_k: int = 5, **_: Any) -> None:
        self.top_k = top_k
        # TODO[indexing-load]: load artifacts produced by the technique's
        # build step (graph, tree, index, etc.) from data/{name}/.

    def answer(self, query: str) -> Answer:
        # TODO[query-embed]: turn the question into the form the technique
        # consumes (embedding, entity-anchor set, etc.).
        # TODO[query-retrieve]: technique-specific retrieval (graph traversal,
        # tree descent, PPR, hybrid fusion, ...).
        # TODO[query-generate]: assemble retrieved context into a prompt and
        # call the LLM. Return Answer(text=..., contexts=..., metadata=...).
        raise NotImplementedError(
            "{name} pipeline not yet implemented — run /implement-paper {name}"
        )
'''

PIPELINE_CONFIG_TEMPLATE = """\
# Pipeline config for {name} — adjust as you implement.
name: {name}
top_k: 5
# Add technique-specific knobs here.
"""

ARXIV_PDF_URL = "https://arxiv.org/pdf/{}"


def scaffold_paper(
    arxiv_id: str, name: str, project_root: Path | None = None
) -> list[Path]:
    """Create papers/<name>_<id>/ + src/rag/techniques/<name>/ + configs/pipelines/<name>.yaml.

    Idempotent — does not overwrite existing files. Returns the list of files created.
    """
    if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id):
        raise ValueError(f"Bad arXiv id (expected NNNN.NNNNN[vN]): {arxiv_id!r}")
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    cls_name = "".join(p.capitalize() for p in name.split("_"))

    root = project_root or Path.cwd()
    paper_dir = root / "papers" / f"{name}_{arxiv_id}"
    technique_dir = root / "src" / "rag" / "techniques" / name
    config_file = root / "configs" / "pipelines" / f"{name}.yaml"

    created: list[Path] = []

    paper_dir.mkdir(parents=True, exist_ok=True)
    meta = paper_dir / "meta.yaml"
    if not meta.exists():
        meta.write_text(PAPER_META_TEMPLATE.format(arxiv_id=arxiv_id, name=name))
        created.append(meta)
    notes = paper_dir / "NOTES.md"
    if not notes.exists():
        notes.write_text(PAPER_NOTES_TEMPLATE.format(name=name))
        created.append(notes)
    pdf = paper_dir / "paper.pdf"
    if not pdf.exists():
        try:
            urlretrieve(ARXIV_PDF_URL.format(arxiv_id), pdf)
            created.append(pdf)
        except Exception as e:
            log.warning("scaffold.pdf-download-failed", arxiv_id=arxiv_id, error=str(e))

    technique_dir.mkdir(parents=True, exist_ok=True)
    init = technique_dir / "__init__.py"
    if not init.exists():
        init.write_text(TECHNIQUE_INIT_TEMPLATE.format(cls_name=cls_name))
        created.append(init)
    pipeline_file = technique_dir / "pipeline.py"
    if not pipeline_file.exists():
        pipeline_file.write_text(
            TECHNIQUE_PIPELINE_TEMPLATE.format(
                arxiv_id=arxiv_id, name=name, cls_name=cls_name
            )
        )
        created.append(pipeline_file)

    config_file.parent.mkdir(parents=True, exist_ok=True)
    if not config_file.exists():
        config_file.write_text(PIPELINE_CONFIG_TEMPLATE.format(name=name))
        created.append(config_file)

    return created
