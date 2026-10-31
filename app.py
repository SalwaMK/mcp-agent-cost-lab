"""Streamlit page for trying questions and viewing their token cost."""

import asyncio
import streamlit as st
from agent import run_agent

# Give the workshop page a clear title and a small prompt box.
st.title("Campus MCP Agent Cost Lab")
st.write("Ask a general question about your courses and notes, then inspect its cost.")
with st.form("question_form"):
    question = st.text_area(
        "Your question",
        value="What's due this week and do my notes cover it?",
        placeholder="For example: Help me plan this week using my deadlines and course notes.",
        height=100,
    )
    ask = st.form_submit_button("Ask the agent")

if ask:
    if not question.strip():
        st.warning("Type a question first.")
        st.stop()
    try:
        # Run the same MCP agent as client.py and show friendly errors in the page.
        with st.spinner("Asking the agent..."):
            report = asyncio.run(run_agent(question))
    except Exception as error:
        st.error(f"Could not finish: {error}")
    else:
        # Put the answer first so students can compare it with the measurements.
        st.subheader("Answer")
        st.write(report["answer"])

        # Highlight the totals that matter most in a cost experiment.
        a, b, c, d = st.columns(4)
        a.metric("Model calls", len(report["steps"]))
        b.metric("Input tokens · blue", report["input_tokens"])
        c.metric("Output tokens · orange", report["output_tokens"])
        d.metric("Estimated cost", f"${report['cost']:.6f}")
        st.metric("Model latency", f"{report['latency']:.2f} seconds")

        # Give each call its own row so the two token counts are easy to compare.
        chart_rows = [{
            "Call": f"Call {row['step']}",
            "Input tokens": row["input_tokens"],
            "Output tokens": row["output_tokens"],
        } for row in report["steps"]]

        # Use separate colors and horizontal bars for a quick comparison.
        st.subheader("Usage by call")
        st.dataframe(report["steps"], hide_index=True, width="stretch",
                     column_config={
                         "step": st.column_config.NumberColumn("Call"),
                         "input_tokens": st.column_config.NumberColumn("Input tokens"),
                         "output_tokens": st.column_config.NumberColumn("Output tokens"),
                         "latency": st.column_config.NumberColumn("Seconds", format="%.2f"),
                         "tools": st.column_config.TextColumn("Tool called"),
                     })
        st.bar_chart(chart_rows, x="Call", y=["Input tokens", "Output tokens"],
                     color=["#3977F6", "#F28E2B"], horizontal=True,
                     stack=False, height=260)
        st.caption("Blue bars show input tokens; orange bars show output tokens. ")
