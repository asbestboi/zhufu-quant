"""Summarize a py-spy profile: the functions where the time goes.

    py-spy record --format raw -o profile.txt -- python scripts/bench/bench.py --case stress
    python scripts/bench/top_functions.py profile.txt --within run_event

Reads py-spy's "raw" output (one collapsed stack per line, followed by its sample count)
and prints the top functions by *self* time (samples where the function itself was
running) and by *total* time (samples where it was anywhere on the stack). With
``--within``, only stacks passing through that function are counted, which leaves out
setup such as data loading.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

# A py-spy frame looks like "function (path/to/file.py:123)"; the line number is dropped
# so that all samples of one function add up.
FRAME = re.compile(r"^(?P<function>.*?) \((?P<file>[^():]*)(?::\d+)?\)$")


def shorten(frame: str) -> str:
    match = FRAME.match(frame.strip())
    if not match:
        return frame.strip()
    file = match["file"]
    for marker in ("/src/qr/", "/site-packages/", "/lib/python3"):
        if marker in file:
            file = file.split(marker, 1)[1]
            break
    return f"{match['function']} ({file})"


def summarize(path: Path, within: str | None) -> tuple[int, Counter[str], Counter[str]]:
    total_samples = 0
    own: Counter[str] = Counter()
    anywhere: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        stack, _, count = line.rpartition(" ")
        if not stack or not count.isdigit():
            continue
        frames = [shorten(f) for f in stack.split(";")]
        if within and not any(f.startswith(f"{within} ") for f in frames):
            continue
        samples = int(count)
        total_samples += samples
        own[frames[-1]] += samples
        for name in set(frames):
            anywhere[name] += samples
    return total_samples, own, anywhere


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("profile", type=Path)
    parser.add_argument("--within", help="only count stacks through this function")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    samples, own, anywhere = summarize(args.profile, args.within)
    print(f"{samples} samples" + (f" inside {args.within}" if args.within else ""))
    for title, counter in (("self time", own), ("total time", anywhere)):
        print(f"\nTop {args.top} functions by {title}:")
        for name, count in counter.most_common(args.top):
            print(f"  {count / samples:6.1%}  {name}")


if __name__ == "__main__":
    main()
