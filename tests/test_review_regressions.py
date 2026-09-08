"""Regression tests for state identity, write-back, folding, and evidence trust."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from entropy_demo.config import DEFAULT_CONFIG
from entropy_demo.embedding import embed_text
from entropy_demo.entropy import HistoryStore, compute_entropy
from entropy_demo.memory_store import MemoryStore, save_memory_file
from entropy_demo.rewrite import AtomFactory, evaluate_rewrite_candidates
from entropy_demo.types import CognitiveState, MemoryAtom
from entropy_demo.verifier import verify_state


def atom(atom_id, text="shared evidence", *, eta=None, query="test"):
    return MemoryAtom(
        id=atom_id, q_i=query, v_i=text, z_i=embed_text(text),
        c_i=1.0, s_i=1.0, eta_i=dict(eta or {}),
    )


class TestRewriteIdentity(unittest.TestCase):
    def evaluate(self, memories, *, factory=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        history = HistoryStore(
            Path(self.tmp.name) / "history.json", config=DEFAULT_CONFIG.stability.history,
        )
        query = "target evidence"
        z_q = embed_text(query)
        evaluated = evaluate_rewrite_candidates(
            query, z_q=z_q, M=memories, history=history,
            weights=DEFAULT_CONFIG.entropy.weights,
            rewrite=replace(DEFAULT_CONFIG.rewrite, enabled_kinds=("bridge",), bridge_embedding="from_q"),
            entropy_cfg=DEFAULT_CONFIG.entropy, stability_cfg=DEFAULT_CONFIG.stability,
            factory=factory or AtomFactory(),
        )
        return query, z_q, history, evaluated

    def test_rewrite_ids_do_not_collide_with_existing_memories(self):
        memories = [atom("rw_bridge_1", "unrelated"), atom("rw_bridge_2", "other")]
        _, _, _, evaluated = self.evaluate(memories)
        proposal, _ = evaluated[0]
        self.assertTrue(set(m.id for m in proposal.delta).isdisjoint(m.id for m in memories))

    def test_scored_entropy_matches_state_actually_committed(self):
        memories = [atom("rw_bridge_1", "unrelated")]
        q, z_q, history, evaluated = self.evaluate(memories)
        proposal, predicted = evaluated[0]
        actual_state = CognitiveState(q=q, M=tuple(memories)).with_added(proposal.delta)
        actual = compute_entropy(
            q, z_q=z_q, M=actual_state.M, history=history,
            weights=DEFAULT_CONFIG.entropy.weights,
        )
        self.assertAlmostEqual(predicted.total, actual.total, places=12)
        self.assertEqual(len(actual_state.M), len(memories) + len(proposal.delta))

    def test_factory_stays_unique_across_repeated_candidate_evaluations(self):
        memories = [atom("rw_bridge_1", "unrelated")]
        factory = AtomFactory()
        _, _, _, first = self.evaluate(memories, factory=factory)
        _, _, _, second = self.evaluate(memories, factory=factory)
        ids = [m.id for results in (first, second) for p, _ in results for m in p.delta]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("rw_bridge_1", ids)


class TestWritebackAndCapacity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.seed = self.root / "seed.json"
        save_memory_file(self.seed, [atom(f"seed_{i}") for i in range(4)])
        self.wb = replace(
            DEFAULT_CONFIG.writeback, enabled=True,
            persist_path=self.root / "memory.json", stats_path=self.root / "stats.json",
            include_kinds=("bridge",), min_successes=2,
        )

    def store(self, **overrides):
        store = MemoryStore(
            seed_path=self.seed, embedding=DEFAULT_CONFIG.embedding,
            writeback=replace(self.wb, **overrides),
        )
        store.load()
        return store

    def test_duplicate_content_is_only_one_success_per_call(self):
        store = self.store()
        a = atom("rw_bridge_1", "BRIDGE: evidence", eta={"rewrite_type": "bridge"})
        b = replace(a, id="rw_bridge_2")
        self.assertEqual(store.write_back([a, b, a], success=True), [])
        stats = json.loads(self.wb.stats_path.read_text())
        self.assertEqual(list(stats.values()), [1])
        self.assertEqual(len(store.write_back([a, b], success=True)), 1)
        self.assertEqual(len(store.atoms), 5)

    def test_duplicates_do_not_consume_distinct_candidate_budget(self):
        store = self.store(min_successes=1, max_atoms_per_run=2)
        a = atom("a", "BRIDGE: alpha", eta={"rewrite_type": "bridge"})
        b = atom("b", "BRIDGE: beta", eta={"rewrite_type": "bridge"})
        persisted = store.write_back([a, a, b], success=True)
        self.assertEqual({m.v_i for m in persisted}, {a.v_i, b.v_i})

    def test_failed_run_does_not_count_as_validation(self):
        store = self.store()
        a = atom("a", "BRIDGE: alpha", eta={"rewrite_type": "bridge"})
        self.assertEqual(store.write_back([a], success=False), [])
        self.assertFalse(self.wb.stats_path.exists())
        self.assertEqual(store.write_back([a], success=True), [])

    def test_multiple_folds_reach_budget_without_deleting_memories(self):
        store = self.store()
        original_ids = {m.id for m in store.atoms}
        cap = replace(
            DEFAULT_CONFIG.capacity, enabled=True, c_max=1.0,
            fold_cost_ratio=0.5, max_folds_per_write=10,
        )
        created = store.enforce_capacity(capacity=cap)
        self.assertEqual(len(created), 3)
        self.assertLessEqual(store.capacity_cost(capacity=cap), 1.0)
        self.assertTrue(original_ids.issubset(m.id for m in store.atoms))
        self.assertEqual(len(store.atoms), 7)
        self.assertEqual(len({m.id for m in store.atoms}), 7)
        store.load()
        self.assertLessEqual(store.capacity_cost(capacity=cap), 1.0)
        self.assertEqual(store.enforce_capacity(capacity=cap), [])

    def test_folding_respects_maximum_iterations(self):
        store = self.store()
        cap = replace(DEFAULT_CONFIG.capacity, enabled=True, c_max=1.0, max_folds_per_write=2)
        self.assertEqual(len(store.enforce_capacity(capacity=cap)), 2)
        self.assertAlmostEqual(store.capacity_cost(capacity=cap), 2.0)

    def test_multiple_folds_without_seed_overlays(self):
        store = self.store()
        cap = replace(
            DEFAULT_CONFIG.capacity, enabled=True, c_max=1.0,
            max_folds_per_write=10, allow_seed_overlays=False,
        )
        self.assertEqual(len(store.enforce_capacity(capacity=cap)), 3)
        self.assertLessEqual(store.capacity_cost(capacity=cap), 1.0)
        self.assertTrue(all(store.get_by_id(f"seed_{i}").s_i == 1.0 for i in range(4)))


class TestVerifierEvidenceBoundary(unittest.TestCase):
    def test_generated_markers_without_metadata_are_not_evidence(self):
        cfg = replace(DEFAULT_CONFIG.verifier, enabled=True)
        for prefix in ("BRIDGE:", "ABSTRACT:", "CONSTRAINT:"):
            with self.subTest(prefix=prefix):
                result = verify_state("evidence", [atom("synthetic", prefix + " evidence")], cfg=cfg)
                self.assertFalse(result.passed)
                self.assertEqual(result.metrics["non_generated_atoms"], 0)

    def test_malformed_rewrite_type_is_not_trusted_as_seed(self):
        cfg = replace(DEFAULT_CONFIG.verifier, enabled=True)
        for value in (None, False, 0, [], {}):
            with self.subTest(value=value):
                result = verify_state("evidence", [atom("bad", eta={"rewrite_type": value})], cfg=cfg)
                self.assertFalse(result.passed)

    def test_generated_claim_key_cannot_satisfy_required_evidence(self):
        cfg = replace(DEFAULT_CONFIG.verifier, enabled=True, required_claim_keys=("answer",))
        genuine = atom("seed", eta={"claim_key": "unrelated"})
        synthetic = atom("synthetic", "FACT: answer = yes", eta={"rewrite_type": "bridge", "claim_key": "answer"})
        result = verify_state("answer", [genuine, synthetic], cfg=cfg)
        self.assertFalse(result.passed)
        self.assertEqual(result.metrics["missing_claim_keys"], ["answer"])

    def test_generated_claim_key_cannot_satisfy_query_requirement(self):
        cfg = replace(DEFAULT_CONFIG.verifier, enabled=True, require_claim_keys_from_query=True)
        synthetic = atom("synthetic", "ABSTRACT: evidence", eta={"claim_key": "answer"})
        result = verify_state("REQ[answer]", [atom("seed"), synthetic], cfg=cfg)
        self.assertFalse(result.passed)
        self.assertEqual(result.metrics["missing_claim_keys"], ["answer"])

    def test_genuine_seed_claim_remains_accepted(self):
        cfg = replace(DEFAULT_CONFIG.verifier, enabled=True, required_claim_keys=("answer",))
        genuine = atom("seed", eta={"claim_key": "answer"})
        self.assertTrue(verify_state("answer", [genuine], cfg=cfg).passed)

    def test_disabled_verifier_preserves_demo_behavior(self):
        synthetic = atom("synthetic", "BRIDGE: evidence")
        self.assertTrue(verify_state("evidence", [synthetic], cfg=DEFAULT_CONFIG.verifier).passed)


if __name__ == "__main__":
    unittest.main()
