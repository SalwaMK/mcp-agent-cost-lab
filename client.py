"""Command-line entry point for asking the campus agent a question."""

import asyncio
import sys
from agent import run_agent


async def main():
    # Use the command-line question, or ask interactively when it is missing.
    question = " ".join(sys.argv[1:]) or input("Ask the campus assistant: ")
    try:
        report = await run_agent(question)
    except Exception as error:
        print(f"Could not finish: {error}")
        return

    # Print one compact row for each call to the model.
    print("Step | Input | Output | Seconds | Tool")
    for row in report["steps"]:
        print(f"{row['step']:>4} | {row['input_tokens']:>5} | "
              f"{row['output_tokens']:>6} | {row['latency']:>7.2f} | {row['tools']}")
    print(f"Totals: {len(report['steps'])} calls, {report['input_tokens']} input, "
          f"{report['output_tokens']} output tokens, {report['latency']:.2f}s, "
          f"${report['cost']:.6f}")
    print(f"\nAnswer: {report['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
