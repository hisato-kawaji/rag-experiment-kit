from rag.techniques.graphrag.extractor import EntityRecord, RelationRecord
from rag.techniques.graphrag.graph import build_graph


def test_dedup_entities_by_lowercased_name():
    ents = [
        EntityRecord(name="Alice", type="PERSON", description="A scientist."),
        EntityRecord(name="alice", type="PERSON", description="A researcher."),
        EntityRecord(name="Bob", type="PERSON", description="An engineer."),
    ]
    rels = [
        RelationRecord(source="Alice", target="Bob", description="colleagues", weight=2.0)
    ]
    g = build_graph(ents, rels)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    edge = g.edges["Alice", "Bob"]
    assert edge["weight"] == 2.0
    # both descriptions kept on the merged Alice node
    assert len(g.nodes["Alice"]["descriptions"]) == 2


def test_relation_with_unknown_endpoint_creates_stub():
    ents = [EntityRecord(name="Alice", type="PERSON", description="")]
    rels = [
        RelationRecord(source="Alice", target="Carol", description="knows", weight=1.0)
    ]
    g = build_graph(ents, rels)
    assert g.number_of_nodes() == 2
    assert "Carol" in g.nodes
    assert g.nodes["Carol"]["type"] == "OTHER"


def test_aggregates_edge_weights():
    ents = [
        EntityRecord(name="A", type="CONCEPT", description=""),
        EntityRecord(name="B", type="CONCEPT", description=""),
    ]
    rels = [
        RelationRecord(source="A", target="B", description="x", weight=1.5),
        RelationRecord(source="B", target="A", description="y", weight=2.5),
    ]
    g = build_graph(ents, rels)
    assert g.number_of_edges() == 1
    assert g.edges["A", "B"]["weight"] == 4.0
