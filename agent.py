"""Run the Groq agent and return its answer and usage measurements."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# GPT OSS 20B supports function calling and costs $0.075/$0.30 per million tokens.
MODEL = "openai/gpt-oss-20b"
PRICE_INPUT = 0.075
PRICE_OUTPUT = 0.30
MAX_STEPS = 6
SYSTEM_PROMPT = """You help students understand fictional campus deadlines and notes.
Use search_notes whenever the question asks what the notes say or whether they cover something.
Use get_deadlines for each course supported by the question or note search.
Base claims on tool results; do not invent a course, deadline, or note.
If a question says 'this week', use the weekdays written in the notes and do not guess calendar dates.
Explain which note lines support your answer, and say when the notes do not give enough detail."""


async def run_agent(question: str) -> dict:
    # Load the Groq key from the root .env file.
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Missing GROQ_API_KEY. Add it to .env first.")

    # Start this folder's MCP server as a separate stdio process.
    server = StdioServerParameters(command=sys.executable,
                                   args=[str(Path(__file__).with_name("server.py"))])
    groq = Groq()
    # The system prompt tells the model when to consult each MCP tool.
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    steps = []

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            # Fail clearly instead of waiting forever if the server cannot start.
            await asyncio.wait_for(session.initialize(), timeout=10)
            # Convert each MCP tool description to Groq's function-tool format.
            available = await session.list_tools()
            tools = [{"type": "function", "function": {
                "name": tool.name, "description": tool.description or "",
                "parameters": tool.inputSchema}}
                for tool in available.tools]

            for number in range(1, MAX_STEPS + 1):
                # The whole message history is resent, so input tokens can grow each call.
                # Time the request and store the token counts returned by Groq.
                started = time.perf_counter()
                reply = groq.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools,
                    max_completion_tokens=700)
                latency = time.perf_counter() - started
                message = reply.choices[0].message
                calls = message.tool_calls or []
                steps.append({"step": number, "input_tokens": reply.usage.prompt_tokens,
                              "output_tokens": reply.usage.completion_tokens,
                              "latency": latency,
                              "tools": ", ".join(c.function.name for c in calls) or "—"})

                # No tool calls means the model has finished its answer.
                if not calls:
                    answer = message.content or "(The model returned an empty answer.)"
                    break

                # Preserve the assistant's tool request before adding tool results.
                messages.append({"role": "assistant", "content": message.content,
                                 "tool_calls": [c.model_dump(exclude_none=True) for c in calls]})
                for call in calls:
                    try:
                        # Run the requested MCP function and return its text to Groq.
                        result = await session.call_tool(
                            call.function.name, json.loads(call.function.arguments))
                        text = "\n".join(item.text for item in result.content
                                         if hasattr(item, "text"))
                        if result.isError:
                            text = "Tool error: " + (text or "tool failed")
                    except Exception as error:
                        text = f"Tool error: {error}"
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": text})
            else:
                answer = "Stopped after six model calls. Try a shorter question."

    # Sum provider token counts and estimate the API charge in US dollars.
    input_total = sum(step["input_tokens"] for step in steps)
    output_total = sum(step["output_tokens"] for step in steps)
    return {"answer": answer, "steps": steps, "input_tokens": input_total,
            "output_tokens": output_total, "latency": sum(s["latency"] for s in steps),
            "cost": (input_total * PRICE_INPUT + output_total * PRICE_OUTPUT) / 1_000_000}
