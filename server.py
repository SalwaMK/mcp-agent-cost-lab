"""Two simple tools for fake campus course information."""

from pathlib import Path
from mcp.server.fastmcp import FastMCP

# FastMCP reads these function docstrings to describe the tools to the model.
mcp = FastMCP("campus-assistant")

# Keep the example deadlines in code so students can easily change them.
DEADLINES = {
    "CS101": "Project proposal: Friday; quiz: next Tuesday.",
    "MATH201": "Homework 4: Thursday.",
    "BIO110": "Field trip forms: Friday; lab report: next Monday.",
    "HIST205": "Reading response: Wednesday.",
    "ENG150": "Draft essay: next Thursday.",
    "PHYS120": "Problem set 6: Friday.",
}


@mcp.tool()
def get_deadlines(course: str) -> str:
    """Get deadlines for a course. Example: get_deadlines("CS101")."""
    # Make the lookup case-insensitive and explain when the course is unknown.
    return DEADLINES.get(course.upper(), f"I couldn't find deadlines for {course}.")


@mcp.tool()
def search_notes(query: str) -> str:
    """Search notes by keyword. Example: search_notes("quiz")."""
    # Find notes.txt beside this server, even when launched from another folder.
    notes = Path(__file__).resolve().parent / "notes.txt"
    # Keep matching case-insensitive and limit the result to three short lines.
    matches = [line for line in notes.read_text().splitlines()
               if query.casefold() in line.casefold()]
    return "\n".join(matches[:3]) or f"I couldn't find notes about {query}."


if __name__ == "__main__":
    # MCP uses stdin and stdout to talk to the client.
    mcp.run(transport="stdio")
