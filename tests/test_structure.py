"""Property tests for true-tree construction."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from orgchart.generation.catalog import ARCHETYPES, DEPARTMENTS
from orgchart.generation.structure import build_tree
from orgchart.ladder import Level

ARCHETYPE_KEYS = sorted(ARCHETYPES)
REGIONS = ("ANZ", "APAC", "EMEA")


def _check_tree(tree, true_size):
    employees = tree.employees
    nodes = tree.nodes
    roots = [n for n in employees if n.parent is None]
    assert len(roots) == 1 and roots[0].role_type == "ceo" and roots[0].level == Level.CEO

    # Every employee reachable from the CEO exactly once => connected and acyclic.
    depth = tree.depth_of()
    assert set(depth) == {n.nid for n in employees}

    for node in employees:
        if node.parent is None:
            continue
        parent = nodes[node.parent]
        assert parent.level > node.level, "levels must strictly decrease along reporting lines"
        assert node.nid in parent.children
        assert parent.role_type != "board"
    for node in nodes:
        for child in node.children:
            assert nodes[child].parent == node.nid
        if node.role_type == "board":
            assert node.parent is None and not node.children and node.level is None
        if node.role_type == "ic":
            assert not node.children and node.level <= Level.LEAD
        if node.role_type in ("manager", "head"):
            assert node.level >= Level.LEAD

    # Size is the requested size, plus at most a few executive assistants for C-levels.
    assert true_size <= len(employees) <= true_size + 12


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    archetype=st.sampled_from(ARCHETYPE_KEYS),
    true_size=st.integers(101, 2500),
)
def test_tree_invariants(seed, archetype, true_size, config):
    tree = build_tree(np.random.default_rng(seed), config, archetype, true_size, REGIONS)
    _check_tree(tree, true_size)


def test_frontline_team_sizes_respect_department_caps(config):
    for seed in range(5):
        tree = build_tree(np.random.default_rng(seed), config, "retail", 1500, REGIONS)
        for node in tree.employees:
            ic_reports = [c for c in node.children if tree.nodes[c].role_type == "ic"]
            if node.role_type in ("manager", "head") and node.level <= Level.MANAGER:
                assert len(ic_reports) <= DEPARTMENTS[node.dept].ic_team_cap + 3


def test_regional_company_scopes_first_split(config):
    found = False
    for seed in range(40):
        tree = build_tree(np.random.default_rng(seed), config, "retail", 1200, REGIONS)
        if tree.scope_kind == "regional" and tree.scopes:
            found = True
            dept, scopes = next(iter(tree.scopes.items()))
            head = next(n for n in tree.employees if n.role_type == "head" and n.dept == dept)
            child_scopes = {tree.nodes[c].scope for c in head.children}
            assert child_scopes <= set(scopes) and len(child_scopes) >= 2
            break
    assert found, "expected at least one regional retail company across 40 seeds"


def test_tree_building_is_deterministic(config):
    a = build_tree(np.random.default_rng(42), config, "saas", 800, REGIONS)
    b = build_tree(np.random.default_rng(42), config, "saas", 800, REGIONS)
    assert [(n.parent, n.level, n.dept, n.specialty) for n in a.nodes] == [
        (n.parent, n.level, n.dept, n.specialty) for n in b.nodes
    ]
