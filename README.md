# MCP Agent Cost Lab

This is a small Python MCP server and Groq agent for answering questions about fake campus courses. The client measures each model call, and the Streamlit page displays the answer, token counts, latency, and estimated cost.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Groq API key to `.env` as `GROQ_API_KEY=...`.

## Run

Ask from the terminal:

```bash
python client.py "What's due for CS101?"
python client.py "What's due this week and do my notes cover it?"
```

Open the visual report:

```bash
streamlit run app.py
```

The server offers two tools: `get_deadlines(course)` and `search_notes(query)`. For broad deadline questions, the client gathers notes and deadline context for every course before asking Groq, so the model can answer without asking which course; since the notes contain weekdays rather than calendar dates, it will say when the exact week is unclear. Change the tool descriptions or notes, then compare calls, tokens, latency, and estimated cost in the app.

## Workshop activity

Run both questions, then change a tool description or shorten the search results and compare the reports. Record your measurements:

| Question | Calls | Input tokens | Output tokens | Latency | Est. cost |
|---|---:|---:|---:|---:|---:|
| What's due for CS101? |  |  |  |  |  |
| What's due this week and do my notes cover it? |  |  |  |  |  |

If 10,000 students ask the multi-step question every day, what is the estimated monthly cost? Multiply the per-question cost by 10,000 and the number of days in the month.

## Troubleshooting

- **Missing key:** check that `.env` is in this folder and contains a valid `GROQ_API_KEY`.
- **Wrong Python version:** run `python --version`; use Python 3.10 or newer.
- **Server issue:** activate `.venv`, install the requirements, and run `python server.py` only for diagnosis; the client normally starts it over stdio.
