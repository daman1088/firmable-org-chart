import numpy as np
import pytest

from orgchart.generation import noise
from orgchart.generation.catalog import ARCHETYPES, DEPARTMENTS
from orgchart.generation.structure import Node, build_tree
from orgchart.generation.titles import assign_canonical_titles, canonical_title, exec_abbreviation


@pytest.fixture
def titled_tree(config):
    tree = build_tree(np.random.default_rng(3), config, "saas", 600, ("ANZ", "EMEA"))
    assign_canonical_titles(np.random.default_rng(4), tree, ARCHETYPES["saas"])
    return tree


def test_every_department_and_level_renders(config):
    rng = np.random.default_rng(0)
    for key, spec in DEPARTMENTS.items():
        for specialty in [None, *spec.specialties]:
            for level in range(0, 5):
                node = Node(0, None, level, key, "ic", specialty=specialty or next(iter(spec.specialties)))
                title = canonical_title(rng, node, [node], ARCHETYPES["saas"])
                assert title and "{" not in title, (key, level, title)
            for level in range(4, 11):
                node = Node(0, None, level, key, "manager", specialty=specialty)
                title = canonical_title(rng, node, [node], ARCHETYPES["saas"])
                assert title and "{" not in title, (key, level, title)
                if specialty is None:
                    # Guard against "Head of Sales, Sales": the function repeated as a qualifier.
                    assert not (title.endswith(f", {spec.func}") and title.count(spec.func) > 1), title
            for level in range(5, 12):
                node = Node(0, None, level, key, "head")
                assert "{" not in canonical_title(rng, node, [node], ARCHETYPES["saas"])


def test_titles_have_no_placeholder_or_doubled_words(titled_tree):
    for node in titled_tree.nodes:
        assert node.title
        assert "{" not in node.title
        words = node.title.split()
        assert all(a != b for a, b in zip(words, words[1:], strict=False)), node.title


def test_stale_rendering_keeps_role_noun(titled_tree, config):
    archetype = ARCHETYPES["saas"]
    rng = np.random.default_rng(1)
    for node in titled_tree.nodes:
        if node.role_type == "ic" and node.level == 3 and node.role_noun:
            lower = canonical_title(rng, node, titled_tree.nodes, archetype, level=2, allow_decorations=False)
            assert node.role_noun.split()[-1] in lower


def test_exec_abbreviation():
    assert exec_abbreviation("Chief Financial Officer") == "CFO"
    assert exec_abbreviation("Co-Founder & CTO") == "CTO"
    assert exec_abbreviation("Chief People Officer") == "CPO"


def _ctx(config, holdout):
    return noise.TitleContext(cfg=config.noise, holdout=holdout, small_company=False, company_short="Acme",
                              render_at_level=lambda n, lvl: n.title)


def test_holdout_transforms_only_when_enabled(titled_tree, config):
    rng = np.random.default_rng(5)
    for _ in range(5):
        for node in titled_tree.nodes:
            _, tags = noise.observe_title(rng, node, _ctx(config, holdout=False))
            assert not any(t.startswith("holdout:") for t in tags)
    seen = set()
    for _ in range(5):
        for node in titled_tree.nodes:
            seen.update(noise.observe_title(rng, node, _ctx(config, holdout=True))[1])
    assert any(t.startswith("holdout:") for t in seen)


def test_title_transforms():
    rng = np.random.default_rng(0)
    assert noise._drop_qualifier(rng, "Engineering Manager, Payments") == "Engineering Manager"
    assert noise._reorder_qualifier(rng, "Engineering Manager, Payments") == "Payments Engineering Manager"
    assert noise._reorder_qualifier(rng, "Director of Engineering") == "Engineering Director"
    assert noise._parenthesise(rng, "Software Engineer, Backend") == "Software Engineer (Backend)"
    out = noise._apply_rules(rng, "Senior Vice President, Sales", noise.TERM_ABBREVIATIONS, 1.0)
    assert out == "SVP, Sales"
    # Whole-word only: "and" inside a word must not be touched.
    assert noise._apply_rules(rng, "Candidate Manager", [("and", ("&",))], 1.0) == "Candidate Manager"


def test_coverage_guarantees_minimum_and_keeps_board(config):
    tree = build_tree(np.random.default_rng(9), config, "consulting", 115, ())
    observed = noise.simulate_coverage(np.random.default_rng(9), tree, coverage=0.70, cfg=config.coverage,
                                       min_observed=110)
    employees_observed = [n for n in tree.employees if n.nid in observed]
    assert len(employees_observed) >= 110
    assert all(n.nid in observed for n in tree.nodes if n.role_type == "board")


def test_middle_managers_missing_more_often_than_executives(config):
    rng = np.random.default_rng(11)
    exec_missing = mid_missing = exec_total = mid_total = 0
    for seed in range(15):
        tree = build_tree(np.random.default_rng(seed), config, "saas", 1500, ())
        observed = noise.simulate_coverage(rng, tree, 0.8, config.coverage, 100)
        for n in tree.employees:
            if n.level >= 9:
                exec_total += 1
                exec_missing += n.nid not in observed
            elif 4 <= n.level <= 8:
                mid_total += 1
                mid_missing += n.nid not in observed
    assert mid_missing / mid_total > exec_missing / exec_total


def test_nearest_observed_manager_skips_unobserved(config):
    tree = build_tree(np.random.default_rng(2), config, "saas", 300, ())
    observed = {n.nid for n in tree.nodes} - {tree.root.children[0]}
    nearest = noise.nearest_observed_manager(tree, observed)
    skipped = tree.root.children[0]
    for child in tree.nodes[skipped].children:
        assert nearest[child] == tree.root.nid
