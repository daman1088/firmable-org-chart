"""Org chart rendering for the explorer: node selection, grouping and Graphviz DOT.

Edges come only from the predictions shown (the view columns):

* DIRECT: solid green, labelled with its probability. The exact manager.
* ANCESTOR: dashed blue, labelled "ancestor relationship", at least two ranks apart. The named person is somewhere
  in the reporting chain above; whether they are the direct manager is not established, so it is never drawn as a
  reporting line, and people grouped under them sit in a box headed "N employees linked through hierarchy /
  direct manager not identified".
* AMBIGUOUS: dotted orange "?" to each candidate.

To keep the chart readable, only people who anchor structure (they have predicted reports or are on the focus
person's chain) are drawn as individual boxes. Everyone else is listed in one box per (parent, kind of claim),
and people with no claim are not drawn at all (they are in the review queue).
"""

from __future__ import annotations

import html
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from app_data import STATUS_COLORS

VIEWS = ("Managers and leaders", "Around an employee", "One department", "Whole company")
LAYOUTS = ("Auto", "Top-down", "Left-to-right")
DEFAULT_LEADERS = 20
MAX_INDIVIDUALS = 120  # individual boxes in department / whole-company views; the rest are grouped or counted
GROUP_ROWS = 6  # names listed per group box before "+ N more"
MAX_WIDTH_PX = 1150  # Auto layout switches a branch to left-to-right when top-down would be wider than this
CARD_WIDTH_PX = 330  # branches narrower than this share a row (three per row); wider ones get the full width
CHAR_PX, SMALL_CHAR_PX, GROUP_CHAR_PX = 6.6, 5.6, 6.6  # rendered px per character: names, 9 pt info line, lists
EDGE_LABEL_PX = 130  # an "ancestor relationship" label beside the dashed line it labels

DIRECT, ANCESTOR, AMBIGUOUS = "DIRECT", "ANCESTOR", "AMBIGUOUS"
EDGE_COLORS = {DIRECT: "#2e7d32", ANCESTOR: "#1565c0", AMBIGUOUS: "#ef6c00"}
HEADER_FILL = {DIRECT: "#e8f5e9", ANCESTOR: "#e3f2fd", AMBIGUOUS: "#fff3e0"}
MAX_HOPS = 25


@dataclass
class Group:
    id: str
    kind: str
    parents: tuple[str, ...]
    members: list[tuple[str, float | None]]  # (employee_id, probability), most confident first


@dataclass
class Graph:
    people: set[str]
    groups: list[Group]
    edges: list[tuple[str, str, str, float | None]]  # (parent, child node id, kind, probability)
    focus: str | None = None
    not_drawn: int = 0  # people in scope that are neither drawn nor listed in a group
    truncated: int = 0  # individual boxes dropped by the size cap
    continued_from: str | None = None  # set on a card that continues a branch split for width
    continues_in: list[str] = field(default_factory=list)  # people whose teams continue on their own cards
    head: str | None = None  # the person whose team a continued card shows
    labels: dict[int, float] = field(default_factory=dict)  # rank -> width taken by ANCESTOR edge labels, px
    ranks: dict[str, int] = field(default_factory=dict)
    widths: dict[str, float] = field(default_factory=dict)  # estimated rendered width of each box, px

    def max_rank_width(self) -> int:
        return max(Counter(self.ranks.values()).values(), default=0)

    def width(self, direction: str = "TB") -> float:
        """Estimated drawing width: widest level top-down, or the sum of level widths left-to-right."""
        by: dict[int, list[float]] = {}
        for n, r in self.ranks.items():
            by.setdefault(r, []).append(self.widths.get(n, 180.0))
        for p, c, k, _ in self.edges:  # a dashed line skips a rank, and its label takes room there
            if k == ANCESTOR and p in self.ranks:
                by.setdefault(self.ranks[p] + 1, []).append(EDGE_LABEL_PX)
        if not by:
            return 0.0
        if direction == "LR":
            return sum(max(v) for v in by.values()) + 90.0 * (len(by) - 1)
        return max(sum(v) + 30.0 * (len(v) - 1) for v in by.values())

    def node_count(self) -> int:
        return len(self.people) + len(self.groups)


def claim_edges(view: pd.DataFrame) -> pd.DataFrame:
    """One row per drawn claim: (parent, child, kind, p). Uses the view status, so a custom threshold applies."""
    s = view["view_status"]
    d = view[s == DIRECT]
    a = view[(s == ANCESTOR) & view["ancestor_id"].notna()]
    m = view.loc[s == AMBIGUOUS, ["employee_id", "candidate_ids", "confidence"]].explode("candidate_ids")
    parts = [
        pd.DataFrame({"parent": d["view_manager"], "child": d["employee_id"], "kind": DIRECT,
                      "p": d["view_confidence"]}),
        pd.DataFrame({"parent": a["ancestor_id"], "child": a["employee_id"], "kind": ANCESTOR, "p": a["confidence"]}),
        pd.DataFrame({"parent": m["candidate_ids"], "child": m["employee_id"], "kind": AMBIGUOUS,
                      "p": m["confidence"]}),
    ]
    e = pd.concat([p.astype({"parent": object, "child": object}) for p in parts], ignore_index=True)
    e = e[e["parent"].map(lambda x: isinstance(x, str))]
    return e.reset_index(drop=True)


@dataclass
class Structure:
    edges: pd.DataFrame
    primary: dict[str, str]  # child -> DIRECT or ANCESTOR parent
    counts: dict[str, Counter]  # parent -> claims by kind
    below: Counter  # people whose DIRECT/ANCESTOR chain passes through this person

    def chain(self, eid: str) -> list[str]:
        out, node = [], self.primary.get(eid)
        while isinstance(node, str) and node not in out and len(out) < MAX_HOPS:
            out.append(node)
            node = self.primary.get(node)
        return out


def structure(view: pd.DataFrame) -> Structure:
    e = claim_edges(view)
    single = e[e["kind"] != AMBIGUOUS]
    primary = dict(zip(single["child"], single["parent"], strict=True))
    counts: dict[str, Counter] = {}
    for parent, kind in zip(e["parent"], e["kind"], strict=True):
        counts.setdefault(parent, Counter())[kind] += 1
    s = Structure(e, primary, counts, Counter())
    for child in primary:
        s.below.update(s.chain(child))
    return s


def _employees(people: pd.DataFrame) -> pd.DataFrame:
    return people[~people["is_non_employee"]]


def _priority(s: Structure, level: pd.Series):
    def key(eid: str) -> tuple:
        lv = level.get(eid)
        return (-s.below.get(eid, 0), -(lv if pd.notna(lv) else -1), eid)
    return key


def _with_chains(s: Structure, ordered: list[str], limit: int) -> set[str]:
    """Take people in priority order, each with their chain upwards, until `limit` is reached."""
    chosen: list[str] = []
    seen: set[str] = set()
    for eid in ordered:
        if len(chosen) >= limit:
            break
        for x in (eid, *s.chain(eid)):
            if x not in seen:
                seen.add(x)
                chosen.append(x)
    return seen


def _leaders(s: Structure, ordered: list[str], budget: int) -> set[str]:
    """Leaders in priority order (with chains) until boxes drawn, counting each one's team lists, reach `budget`."""
    chosen: set[str] = set()
    cost = 0
    for eid in ordered:
        if cost >= budget:
            break
        for x in (eid, *s.chain(eid)):
            if x not in chosen:
                chosen.add(x)
                cost += 1 + len(s.counts.get(x, {}))  # the person plus one list box per kind of claim
    return chosen


def select(people: pd.DataFrame, view: pd.DataFrame, mode: str, focus: str | None = None,
           department: str | None = None, n_leaders: int = DEFAULT_LEADERS) -> Graph:
    """Choose individual boxes and grouped members for a view, and build the graph."""
    s = structure(view)
    emp = _employees(people)
    level = emp.set_index("employee_id")["level"]
    parents = set(s.edges["parent"])
    key = _priority(s, level)
    truncated = 0

    if mode == "Around an employee" and focus:
        individuals = {focus, *s.chain(focus)}
        mine = s.edges[s.edges["child"] == focus]
        individuals |= set(mine["parent"])  # AMBIGUOUS candidates
        boss = s.primary.get(focus)
        nearby = s.edges[s.edges["parent"].isin({focus} | ({boss} if boss else set()))]
        members = individuals | set(nearby["child"])
    elif mode in ("One department", "Whole company"):
        scope = set(emp["employee_id"]) if mode == "Whole company" else set(
            emp.loc[emp["department"] == department, "employee_id"])
        linked = s.edges[s.edges["child"].isin(scope)]
        anchors = sorted(set(linked["parent"]) | (parents & scope), key=key)
        individuals = _with_chains(s, anchors, MAX_INDIVIDUALS)
        truncated = max(0, len(set(anchors) - individuals))
        members = (scope & (set(linked["child"]) | parents)) | individuals
    else:  # Managers and leaders: the people that anchor the most predicted structure, with their teams grouped
        individuals = _leaders(s, sorted(parents, key=key), n_leaders)
        members = individuals | set(s.edges.loc[s.edges["parent"].isin(individuals), "child"])

    graph = _build(s, individuals, members, focus if mode == "Around an employee" else None)
    graph.truncated = truncated
    graph.widths = _widths(graph, people, s)
    return graph


def _widths(graph: Graph, people: pd.DataFrame, s: Structure) -> dict[str, float]:
    info = people.set_index("employee_id")

    def text(v: object, n: int) -> int:
        return min(len(str(v)) if isinstance(v, str) else 10, n)

    out = {e: 30 + max(CHAR_PX * max(text(info.at[e, "full_name"], 34), text(info.at[e, "title_raw"], 34)),
                       SMALL_CHAR_PX * len(_meta(info.at[e, "level"], s.counts.get(e, Counter()))))
           for e in graph.people}
    for g in graph.groups:
        rows = [text(info.at[e, "full_name"], 22) + text(info.at[e, "title_raw"], 26) + 3 for e, _ in g.members]
        out[g.id] = 60 + GROUP_CHAR_PX * max([*rows[:GROUP_ROWS], *map(len, _group_header(g)), 38])
    return out


def _build(s: Structure, individuals: set[str], members: set[str], focus: str | None) -> Graph:
    e = s.edges[s.edges["parent"].isin(individuals) & s.edges["child"].isin(members | individuals)]
    direct_edges = e[e["child"].isin(individuals)]
    grouped = e[~e["child"].isin(individuals)]
    edges = [(p, c, k, None if pd.isna(q) else float(q)) for p, c, k, q in direct_edges.itertuples(index=False)]

    groups: list[Group] = []
    keyed: dict[tuple, list[tuple[str, float | None]]] = {}
    amb = grouped[grouped["kind"] == AMBIGUOUS].groupby("child")["parent"].agg(lambda x: tuple(sorted(x)))
    for child, cand in amb.items():
        p = grouped.loc[(grouped["child"] == child), "p"].iloc[0]
        keyed.setdefault((AMBIGUOUS, cand), []).append((child, None if pd.isna(p) else float(p)))
    for (parent, kind), rows in grouped[grouped["kind"] != AMBIGUOUS].groupby(["parent", "kind"], sort=True):
        keyed[(kind, (parent,))] = [(c, None if pd.isna(q) else float(q)) for c, q in zip(rows["child"], rows["p"],
                                                                                          strict=True)]
    for i, ((kind, parents), rows) in enumerate(sorted(keyed.items())):
        rows.sort(key=lambda r: (-(r[1] or 0.0), r[0]))
        g = Group(f"group_{i}", kind, parents, rows)
        groups.append(g)
        edges += [(p, g.id, kind, None) for p in parents]

    listed = set(grouped["child"])
    graph = Graph(set(individuals), groups, edges, focus, not_drawn=len(members - individuals - listed))
    graph.ranks = _ranks(graph)
    return graph


def _ranks(graph: Graph) -> dict[str, int]:
    """Approximate Graphviz ranks (longest path, ANCESTOR spans two) to estimate the drawing's width."""
    ups: dict[str, list[tuple[str, int]]] = {}
    for p, c, k, _ in graph.edges:
        ups.setdefault(c, []).append((p, 2 if k == ANCESTOR else 1))
    memo: dict[str, int] = {}

    def rank(node: str, depth: int = 0) -> int:
        if node not in memo:
            memo[node] = 0 if depth > MAX_HOPS else max((rank(p, depth + 1) + n for p, n in ups.get(node, [])),
                                                         default=0)
        return memo[node]

    return {n: rank(n) for n in [*graph.people, *(g.id for g in graph.groups)]}


def components(graph: Graph) -> list[Graph]:
    """Split into connected branches (largest first). Branches share no predicted link, so each is drawn alone."""
    nodes = [*graph.people, *(g.id for g in graph.groups)]
    root = {n: n for n in nodes}

    def find(x: str) -> str:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for p, c, _, _ in graph.edges:
        root[find(p)] = find(c)
    by: dict[str, list[str]] = {}
    for n in nodes:
        by.setdefault(find(n), []).append(n)
    groups = {g.id: g for g in graph.groups}
    out = []
    for members in by.values():
        ms = set(members)
        sub = Graph({n for n in ms if n not in groups}, [groups[n] for n in members if n in groups],
                    [e for e in graph.edges if e[1] in ms], graph.focus if graph.focus in ms else None)
        sub.ranks = {n: graph.ranks.get(n, 0) for n in ms}
        sub.widths = {n: graph.widths[n] for n in ms if n in graph.widths}
        out.append(sub)

    def size(g: Graph) -> tuple:
        return (g.focus is None, -(len(g.people) + sum(len(x.members) for x in g.groups)), min(g.people, default=""))
    return sorted(out, key=size)


def _subgraph(graph: Graph, nodes: set[str]) -> Graph:
    groups = [g for g in graph.groups if g.id in nodes]
    nodes = nodes | {p for g in groups for p in g.parents}  # keep every "?" candidate of a grouped AMBIGUOUS claim
    sub = Graph({n for n in nodes if n in graph.people}, groups,
                [e for e in graph.edges if e[0] in nodes and e[1] in nodes],
                graph.focus if graph.focus in nodes else None)
    sub.ranks = {n: graph.ranks.get(n, 0) for n in nodes}
    sub.widths = {n: graph.widths[n] for n in nodes if n in graph.widths}
    return sub


def split(graph: Graph, max_px: float = MAX_WIDTH_PX) -> list[Graph]:
    """Split a branch that is too wide top-down: the top person and their reports on one card, and each report's
    own team continuing on its own card (recursively). Branches around a focus employee are never split."""
    tops = roots(graph)
    if graph.focus or graph.width("TB") <= max_px or len(tops) != 1:
        return [graph]
    top = tops[0]
    down: dict[str, list[str]] = {}
    for p, c, _, _ in graph.edges:
        down.setdefault(p, []).append(c)
    heads = sorted(c for c in set(down.get(top, [])) if c in graph.people and c in down)
    if not heads:
        return [graph]
    taken: set[str] = set()
    parts: list[Graph] = []
    for h in heads:
        stack, team = [h], set()
        while stack:
            x = stack.pop()
            if x not in team and x not in taken:
                team.add(x)
                stack += down.get(x, [])
        taken |= team - {h}
        sub = _subgraph(graph, team)
        # A grouped AMBIGUOUS claim brings its candidates along, which can put the top person back in the team;
        # only recurse on a team that is strictly smaller, or the split never ends.
        for card in split(sub, max_px) if sub.people < graph.people else [sub]:
            card.continued_from = card.continued_from or top
            card.head = card.head or h
            parts.append(card)
    first = _subgraph(graph, set(graph.ranks) - taken)
    first.continues_in = heads
    return [first, *parts]


def roots(graph: Graph) -> list[str]:
    """People at the top of a branch (no drawn claim above them)."""
    children = {c for _, c, _, _ in graph.edges}
    return sorted(n for n in graph.people if n not in children)


def direction_for(graph: Graph, layout: str) -> str:
    if layout == "Top-down":
        return "TB"
    if layout == "Left-to-right":
        return "LR"
    return "LR" if graph.width("TB") > MAX_WIDTH_PX else "TB"


def needs_full_width(graph: Graph, direction: str) -> bool:
    return graph.focus is not None or graph.width(direction) > CARD_WIDTH_PX


# --- DOT ---------------------------------------------------------------------------------------------

def _t(text: object, n: int = 34) -> str:
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    return html.escape(s if len(s) <= n else s[: n - 1] + "…")


def _meta(level: object, c: Counter) -> str:
    """Info line under a person: level and the claims that name them."""
    parts = [f"L{int(level)}" if pd.notna(level) else "level ?"]
    if c[DIRECT]:
        parts.append(f"{c[DIRECT]} direct report{'s' if c[DIRECT] > 1 else ''}")
    if c[ANCESTOR]:
        parts.append(f"{c[ANCESTOR]} employees linked through hierarchy")
    if c[AMBIGUOUS]:
        parts.append(f"candidate for {c[AMBIGUOUS]}")
    return " · ".join(parts)


def _person_label(info: pd.DataFrame, eid: str, s: Structure) -> str:
    row = info.loc[eid]
    meta = _meta(row["level"], s.counts.get(eid, Counter()))
    return (f'<<B>{_t(row["full_name"])}</B><BR/>{_t(row["title_raw"] or "(no title)")}<BR/>'
            f'<FONT POINT-SIZE="9" COLOR="#555555">{html.escape(meta)}</FONT>>')


def _group_header(g: Group) -> list[str]:
    """Header lines of a group box."""
    n = len(g.members)
    return {DIRECT: [f"{n} direct report{'s' if n > 1 else ''}"],
            ANCESTOR: [f"{n} employees linked through hierarchy", "direct manager not identified"],
            AMBIGUOUS: [f"{n} whose manager is one of the “?” links"]}[g.kind]


def _group_label(info: pd.DataFrame, g: Group, max_rows: int = GROUP_ROWS) -> str:
    n = len(g.members)
    header = '<BR ALIGN="LEFT"/>'.join(html.escape(line) for line in _group_header(g))
    rows = [f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" BGCOLOR="{HEADER_FILL[g.kind]}"><B>{header}</B></TD></TR>']
    for eid, p in g.members[:max_rows]:
        prob = f' <FONT COLOR="#666666">{p:.2f}</FONT>' if p is not None else ""
        rows.append(f'<TR><TD ALIGN="LEFT">{_t(info.at[eid, "full_name"], 22)} · '
                    f'{_t(info.at[eid, "title_raw"] or "(no title)", 26)}{prob}</TD></TR>')
    if n > max_rows:
        rows.append(f'<TR><TD ALIGN="LEFT"><I>+ {n - max_rows} more (see Employees tab)</I></TD></TR>')
    return (f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" COLOR="{EDGE_COLORS[g.kind]}" '
            f'BGCOLOR="white">{"".join(rows)}</TABLE>>')


def to_dot(graph: Graph, people: pd.DataFrame, view: pd.DataFrame, direction: str = "TB",
           max_rows: int = GROUP_ROWS) -> str:
    info = people.set_index("employee_id")
    status = view.set_index("employee_id")["view_status"]
    s = structure(view)
    lines = ["digraph org {",
             f'graph [rankdir={direction}, bgcolor="transparent", nodesep=0.3, ranksep=0.45, fontname="Helvetica"];',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#666666", '
             'margin="0.15,0.07"];',
             'edge [arrowhead=none, fontname="Helvetica", fontsize=9];']
    for eid in sorted(graph.people, key=lambda x: (graph.ranks.get(x, 0), x)):
        fill = STATUS_COLORS.get(status.get(eid, "UNRESOLVED"), "#9e9e9e") + "33"
        border = ', penwidth=3, color="#d81b60"' if eid == graph.focus else ""
        lines.append(f'"{eid}" [label={_person_label(info, eid, s)}, fillcolor="{fill}"{border}];')
    for g in graph.groups:
        lines.append(f'"{g.id}" [shape=plain, style="", label={_group_label(info, g, max_rows)}];')
    for parent, child, kind, p in graph.edges:
        if kind == DIRECT:
            if p is None:
                lines.append(f'"{parent}" -> "{child}" [color="{EDGE_COLORS[DIRECT]}", penwidth=2, label="direct", '
                             f'fontcolor="{EDGE_COLORS[DIRECT]}"];')
            else:
                color = "#1b5e20" if p >= 0.9 else "#43a047" if p >= 0.7 else "#9ccc65"
                lines.append(f'"{parent}" -> "{child}" [color="{color}", penwidth={1 + 2 * p:.1f}, '
                             f'label="{p:.2f}", fontcolor="{color}"];')
        elif kind == ANCESTOR:
            # Not a reporting line: dashed, two ranks apart, labelled "ancestor relationship" (direct or higher).
            lines.append(f'"{parent}" -> "{child}" [style=dashed, color="{EDGE_COLORS[ANCESTOR]}", minlen=2, '
                         f'fontcolor="{EDGE_COLORS[ANCESTOR]}", label="ancestor relationship"];')
        else:
            lines.append(f'"{parent}" -> "{child}" [style=dotted, penwidth=1.5, color="{EDGE_COLORS[AMBIGUOUS]}", '
                         f'fontcolor="{EDGE_COLORS[AMBIGUOUS]}", label="?"];')
    lines.append("}")
    return "\n".join(lines)


def nearby(graph: Graph) -> list[str]:
    """Everyone drawn or listed, for the 'go to' picker."""
    return sorted(graph.people | {eid for g in graph.groups for eid, _ in g.members})
