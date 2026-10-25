"""True org-tree construction.

The tree is built top-down: CEO -> executive office -> department heads ->
recursive subtrees. Every subtree is "grown" from a headcount budget, choosing
spans of control and child manager levels so that:

* levels strictly decrease along every reporting line,
* frontline managers run teams of realistic size (department-specific caps),
* senior managers mostly manage managers, occasionally with skip-levels and
  staff ICs (principal engineers, analysts) reporting directly to them,
* regional / divisional companies split scopable departments by region or
  division at the department-head level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from orgchart.config import GenerationConfig
from orgchart.generation.catalog import ARCHETYPES, DEPARTMENTS, OWNERSHIP, Archetype, DeptSpec
from orgchart.generation.rand import chance, pick, weighted_pick
from orgchart.ladder import Level

EXECUTIVE = "executive"
BOARD = "board"


@dataclass
class Node:
    nid: int
    parent: int | None
    level: int | None
    dept: str
    role_type: str  # ceo | head | manager | ic | assistant | chief_of_staff | board
    specialty: str | None = None
    scope: str | None = None
    employment_type: str = "full_time"
    children: list[int] = field(default_factory=list)
    # Filled in by later pipeline stages.
    title: str = ""
    location: str | None = None
    role_noun: str | None = None  # chosen once so stale/inflated renderings keep the same role


@dataclass
class CompanyTree:
    nodes: list[Node]
    scope_kind: str  # none | regional | divisional
    scopes: dict[str, tuple[str, ...]]  # dept -> scopes used at its first split
    departments: list[str]

    @property
    def employees(self) -> list[Node]:
        return [n for n in self.nodes if n.role_type != BOARD]

    @property
    def root(self) -> Node:
        return self.nodes[0]

    def depth_of(self) -> dict[int, int]:
        """Depth below the CEO for every employee (BFS; heads may be re-parented after creation)."""
        depth = {self.root.nid: 0}
        frontier = [self.root.nid]
        while frontier:
            nxt = []
            for nid in frontier:
                for child in self.nodes[nid].children:
                    depth[child] = depth[nid] + 1
                    nxt.append(child)
            frontier = nxt
        return depth

    def bfs_order(self) -> list[Node]:
        order = sorted(self.depth_of().items(), key=lambda kv: (kv[1], kv[0]))
        return [self.nodes[nid] for nid, _ in order]


def _max_subtree(cap: int) -> dict[int, int]:
    """Largest subtree (excluding the manager) a manager at each level should own."""
    m5 = 5 * cap
    m6 = max(140, 3 * m5)
    m7 = max(450, 3 * m6)
    return {4: cap, 5: m5, 6: m6, 7: m7, 8: max(1000, 2 * m7), 9: 2500, 10: 6000, 11: 10**9, 12: 10**9}


def _ceil_level(budget: int, cap: int) -> int:
    """Most senior level that is plausible for someone owning `budget` people."""
    if budget <= cap:
        return 5
    if budget <= max(25, int(2.5 * cap)):
        return 6
    if budget <= 70:
        return 7
    if budget <= 200:
        return 8
    if budget <= 600:
        return 9
    if budget <= 1500:
        return 10
    return 11


class TreeBuilder:
    def __init__(self, rng: np.random.Generator, cfg: GenerationConfig, archetype: Archetype, true_size: int):
        self.rng = rng
        self.cfg = cfg
        self.scfg = cfg.structure
        self.arch = archetype
        self.n_target = true_size
        self.nodes: list[Node] = []

    # ------------------------------------------------------------------ helpers
    def _add(self, parent: Node | None, level: int | None, dept: str, role_type: str, **kw) -> Node:
        node = Node(nid=len(self.nodes), parent=None if parent is None else parent.nid, level=level, dept=dept,
                    role_type=role_type, **kw)
        self.nodes.append(node)
        if parent is not None:
            parent.children.append(node.nid)
        return node

    def _employment_type(self, level: int) -> str:
        if level == Level.INTERN:
            return "intern"
        if chance(self.rng, self.scfg.contractor_rate):
            return "contractor"
        return "full_time"

    def _add_ics(self, manager: Node, n: int, spec: DeptSpec, staff: bool = False) -> None:
        if staff:
            weights = {3: 0.5, 4: 0.5}
        else:
            weights = {lvl: w for lvl, w in spec.ic_level_weights.items() if lvl < manager.level}
        specialties = list(spec.specialties)
        for _ in range(n):
            level = weighted_pick(self.rng, weights)
            specialty = manager.specialty or pick(self.rng, specialties)
            self._add(manager, level, manager.dept, "ic", specialty=specialty, scope=manager.scope,
                      employment_type=self._employment_type(level))

    def _split(self, total: int, m: int, cap: int, alpha: float) -> list[int]:
        """Split `total` people into `m` parts, each in [1, cap]. Higher alpha = more even parts."""
        weights = self.rng.dirichlet(np.full(m, alpha))
        parts = 1 + self.rng.multinomial(total - m, weights)
        while parts.max() > cap:
            i = int(parts.argmax())
            excess = int(parts[i] - cap)
            parts[i] = cap
            for j in np.argsort(parts):
                room = int(cap - parts[j])
                if room <= 0:
                    continue
                take = min(room, excess)
                parts[j] += take
                excess -= take
                if excess == 0:
                    break
        return [int(p) for p in parts]

    # ------------------------------------------------------------------ growth
    def _grow(self, node: Node, budget: int, spec: DeptSpec, scopes: tuple[str, ...] = ()) -> None:
        if budget <= 0:
            return
        level = node.level
        cap = spec.ic_team_cap
        max_sub = _max_subtree(cap)
        if level == Level.MANAGER:
            direct_cap = cap + 3
        elif level == Level.SENIOR_MANAGER:
            direct_cap = cap
        else:
            direct_cap = min(cap, 8)
        if level <= Level.LEAD or (budget <= direct_cap and not scopes):
            self._add_ics(node, budget, spec)
            return

        lo, hi = self.scfg.span[min(level, 12)]
        if scopes:
            n_direct = 0
        elif level >= Level.DIRECTOR:
            n_direct = int(self.rng.choice([0, 0, 0, 1, 1, 2]))
        else:
            n_direct = int(self.rng.integers(0, 3))
        n_direct = min(n_direct, budget // 4)
        rem = budget - n_direct
        child_cap_total = max_sub[level - 1] + 1
        team_target = int(self.rng.integers(max(3, int(0.6 * cap)), cap + 1))
        frontline = rem <= hi * (cap + 1)
        if frontline:  # children are frontline teams: size them like teams, not like spans
            m = math.ceil(rem / (team_target + 1))
        else:
            m = int(self.rng.integers(lo, hi + 1)) - n_direct
        m = max(2, m, math.ceil(rem / child_cap_total), len(scopes))
        m = min(m, rem)
        parts = self._split(rem, m, child_cap_total, alpha=4.0 if frontline else 1.6)

        # Child specialty / scope assignment.
        specialties = list(spec.specialties)
        self.rng.shuffle(specialties)
        for i, part in enumerate(parts):
            sub = part - 1
            if scopes:
                child_scope, child_spec = scopes[i % len(scopes)], None
            else:
                child_scope = node.scope
                child_spec = node.specialty or (specialties[i % len(specialties)] if specialties else None)
            if sub <= 2 and not scopes and (sub == 0 or chance(self.rng, 0.8)):
                # Too small to be a team: these people report to `node` directly.
                ic_weights = {k: v for k, v in spec.ic_level_weights.items() if k < level}
                for _ in range(part):
                    lvl = weighted_pick(self.rng, ic_weights)
                    self._add(node, lvl, node.dept, "ic", specialty=child_spec or pick(self.rng, specialties),
                              scope=child_scope, employment_type=self._employment_type(lvl))
                continue
            preferred = level - 1 if not chance(self.rng, self.scfg.skip_level_probability) else level - 2
            floor = min(lvl for lvl in range(4, 12) if max_sub[lvl] >= sub)
            ceiling = min(level - 1, _ceil_level(sub, cap))
            child_level = max(floor, min(preferred, ceiling))
            child_level = max(Level.LEAD, min(child_level, level - 1))
            child = self._add(node, child_level, node.dept, "manager", specialty=child_spec, scope=child_scope)
            self._grow(child, sub, spec)

        if n_direct:
            self._add_ics(node, n_direct, spec, staff=level >= Level.DIRECTOR)

    # ------------------------------------------------------------------ company
    def _allocate_departments(self, budget: int) -> dict[str, int]:
        n = self.n_target
        include_p = min(1.0, 0.3 + n / 800)
        depts = [d for d in self.arch.dept_weights if d in self.arch.core_depts or chance(self.rng, include_p)]
        base = np.asarray([self.arch.dept_weights[d] for d in depts], dtype=float)
        probs = self.rng.dirichlet(base / base.sum() * 60)
        counts = dict(zip(depts, (int(c) for c in self.rng.multinomial(budget, probs)), strict=True))
        largest = max(counts, key=counts.__getitem__)
        for d in list(counts):
            if d == largest or counts[d] >= 2:
                continue
            if d in self.arch.core_depts:
                need = 2 - counts[d]
                counts[d] += need
                counts[largest] -= need
            else:
                counts[largest] += counts.pop(d)
        return counts

    def _head_level(self, spec: DeptSpec, headcount: int) -> int:
        c_prob = 0.0
        if spec.c_titles and headcount >= 4:
            c_prob = spec.c_propensity * min(1.0, 0.4 + self.n_target / 1500)
            if headcount >= 250:
                c_prob = max(c_prob, 0.8)
        if chance(self.rng, c_prob):
            level = Level.C_LEVEL
        elif headcount >= 250:
            level = Level.SVP
        elif headcount >= 120:
            level = weighted_pick(self.rng, {10: 0.35, 9: 0.65})
        elif headcount >= 40:
            level = weighted_pick(self.rng, {9: 0.45, 8: 0.25, 7: 0.30})
        elif headcount >= 12:
            level = weighted_pick(self.rng, {8: 0.10, 7: 0.60, 6: 0.30})
        else:
            level = weighted_pick(self.rng, {7: 0.25, 6: 0.25, 5: 0.50})
        max_sub = _max_subtree(spec.ic_team_cap)
        floor = min(lvl for lvl in range(5, 12) if max_sub[lvl] >= headcount - 1)
        return int(max(level, floor))

    def _scope_kind(self) -> str:
        r = self.rng.random()
        if r < self.scfg.regional_probability:
            return "regional"
        if self.n_target >= self.scfg.divisional_min_size and r < self.scfg.regional_probability + self.scfg.divisional_probability:
            return "divisional"
        return "none"

    def build(self, regions: tuple[str, ...]) -> CompanyTree:
        rng, scfg, n = self.rng, self.scfg, self.n_target
        ceo = self._add(None, Level.CEO, EXECUTIVE, "ceo")

        n_special = 0
        if n >= scfg.executive_assistant_min_size and chance(rng, scfg.executive_assistant_probability):
            self._add(ceo, Level.INTERMEDIATE, EXECUTIVE, "assistant")
            n_special += 1
        if n >= scfg.chief_of_staff_min_size and chance(rng, scfg.chief_of_staff_probability):
            self._add(ceo, Level.DIRECTOR, EXECUTIVE, "chief_of_staff")
            n_special += 1

        counts = self._allocate_departments(n - 1 - n_special)
        scope_kind = self._scope_kind()
        divisions = tuple(rng.permutation(list(self.arch.divisions))[: int(rng.integers(2, 4))].tolist())

        heads: dict[str, Node] = {}
        scopes_used: dict[str, tuple[str, ...]] = {}
        # Create heads (attached to CEO for now) and grow their subtrees.
        for dept in sorted(counts, key=lambda d: -counts[d]):
            spec = DEPARTMENTS[dept]
            headcount = counts[dept]
            head = self._add(ceo, self._head_level(spec, headcount), dept, "head")
            heads[dept] = head
            scopes: tuple[str, ...] = ()
            if headcount - 1 >= scfg.scope_min_department_size:
                if scope_kind == "regional" and dept in self.arch.regional_depts and len(regions) >= 2:
                    scopes = regions
                elif scope_kind == "divisional" and dept in self.arch.divisional_depts:
                    scopes = divisions
            if scopes:
                scopes_used[dept] = scopes
            self._grow(head, headcount - 1, spec, scopes)

        self._attach_owned_departments(ceo, heads, counts)

        if n >= scfg.exec_assistant_for_c_level_min_size:
            for dept, head in heads.items():
                if head.level == Level.C_LEVEL and chance(rng, scfg.exec_assistant_for_c_level_probability):
                    self._add(head, Level.INTERMEDIATE, EXECUTIVE, "assistant")

        n_board = min(int(rng.poisson(scfg.board_profiles_mean)), scfg.board_profiles_max)
        for _ in range(n_board):
            self._add(None, None, BOARD, "board", employment_type="board")

        return CompanyTree(self.nodes, scope_kind, scopes_used, sorted(counts))

    def _attach_owned_departments(self, ceo: Node, heads: dict[str, Node], counts: dict[str, int]) -> None:
        """Re-parent some department heads under another department's head (e.g. Legal under CFO).

        Only one level of nesting is allowed (an owner is never itself owned), which
        rules out cycles, and the owner must be strictly more senior.
        """
        owned: set[str] = set()
        owners: set[str] = set()
        for dept in sorted(counts, key=lambda d: counts[d]):  # smallest departments first
            if dept in owners:
                continue
            for owner, p in OWNERSHIP.get(dept, ()):
                if owner not in heads or owner in owned or not chance(self.rng, p):
                    continue
                if heads[owner].level <= heads[dept].level:
                    continue
                head = heads[dept]
                ceo.children.remove(head.nid)
                head.parent = heads[owner].nid
                heads[owner].children.append(head.nid)
                owned.add(dept)
                owners.add(owner)
                break


def build_tree(rng: np.random.Generator, cfg: GenerationConfig, archetype_key: str, true_size: int,
               regions: tuple[str, ...]) -> CompanyTree:
    return TreeBuilder(rng, cfg, ARCHETYPES[archetype_key], true_size).build(regions)
