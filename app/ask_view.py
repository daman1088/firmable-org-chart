"""The "Ask the Org Chart" tab: a question box, example questions and the answer. The logic is in ask.py."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ask import AMBIGUOUS_MATCH, ANSWER, INFERRED, NOT_FOUND, NOT_IDENTIFIED, UNSUPPORTED, ask, examples

CALLOUTS = {ANSWER: st.success, INFERRED: st.info, NOT_IDENTIFIED: st.warning, AMBIGUOUS_MATCH: st.warning,
            NOT_FOUND: st.error, UNSUPPORTED: st.error}
KEY = "ask_question"


def _use(question: str) -> None:
    st.session_state[KEY] = question


def render(people: pd.DataFrame, view: pd.DataFrame, everyone: pd.DataFrame, target_name: str) -> None:
    st.markdown(f"Ask about reporting lines in this company. Answers use the model's predictions at the "
                f"**{target_name}** (the claims the chart draws) and normalised job titles. Questions are matched "
                "to three fixed patterns; no language model is used.")
    st.markdown("**Examples** (from this company):")
    for i, (col, q) in enumerate(zip(st.columns(3), examples(people, view), strict=True)):
        col.button(q, key=f"ask_example_{i}", on_click=_use, args=(q,))
    question = st.text_input("Question", key=KEY, placeholder="Who does Jane Smith report to?")
    if not question.strip():
        st.caption("Type a question, or pick an example above.")
        return
    a = ask(question, people, view, everyone)
    CALLOUTS[a.outcome](a.headline)
    if a.rows:
        st.dataframe(pd.DataFrame(a.rows), hide_index=True, width="stretch",
                     column_config={"confidence": st.column_config.NumberColumn(format="%.2f")})
    for note in a.notes:
        st.caption(note)
    st.caption("DIRECT names the exact manager. ANCESTOR names someone above, who may be the direct manager or "
               "someone higher. An inferred department leader comes from job titles only, not from a reporting "
               "claim.")
