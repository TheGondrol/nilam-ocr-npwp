import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import asyncpg

DIRECTORY = Path(__file__).parent


def statements(text: str) -> Iterator[str]:
    """The statements of `text`, split on `;` except inside a `$$ ... $$` block (a DO block)."""
    pending = ""
    for chunk in text.split(";"):
        pending = f"{pending};{chunk}" if pending else chunk
        if pending.count("$$") % 2:
            continue
        lines = [line for line in pending.splitlines() if line.strip() and not line.strip().startswith("--")]
        pending = ""
        if lines:
            yield "\n".join(lines)


async def apply() -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL is not set: point it at the database that holds the orchestration tables", file=sys.stderr)
        return 1
    connection = await asyncpg.connect(url.replace("+asyncpg", ""))
    try:
        for path in sorted(DIRECTORY.glob("*.sql")):
            applied = 0
            for statement in statements(path.read_text(encoding="utf-8")):
                await connection.execute(statement)
                applied += 1
            print(f"applied {path.name}: {applied} statement(s)")
    finally:
        await connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(apply()))
