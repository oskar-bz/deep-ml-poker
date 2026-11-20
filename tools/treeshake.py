"""Remove unreferenced top-level functions/classes/constants and unreferenced methods from a built
single-file submission, repeating until nothing changes. A name counts as referenced if it appears
(as a whole word) anywhere outside its own definition.

python tools/treeshake.py IN.py OUT.py
"""
import ast
import re
import sys

KEEP = {"MyBot", "ContestCore", "Bot", "Action", "act", "on_hand_end", "on_table_start", "core_factory"}


def _uses_outside(src_lines, name, start, end):
    pat = re.compile(r"\b%s\b" % re.escape(name))
    for i, line in enumerate(src_lines):
        if start <= i + 1 <= end:
            continue
        code = line.split("#", 1)[0]
        if pat.search(code):
            return True
    return False


def _candidates(tree):
    """(name, start_line, end_line) for top-level defs/assigns and class methods."""
    out = []
    for n in tree.body:
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.append((n.name, start, n.end_lineno))
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, ast.FunctionDef) and not m.name.startswith("__"):
                        ms = min([m.lineno] + [d.lineno for d in m.decorator_list])
                        out.append((m.name, ms, m.end_lineno))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if len(names) == 1:
                out.append((names[0], start, n.end_lineno))
    return out


def shake(src: str):
    removed = []
    while True:
        tree = ast.parse(src)
        lines = src.split("\n")
        drop = None
        for name, s, e in _candidates(tree):
            if name in KEEP or name.startswith("__"):
                continue
            if not _uses_outside(lines, name, s, e):
                drop = (name, s, e)
                break
        if drop is None:
            return src, removed
        name, s, e = drop
        removed.append((name, e - s + 1))
        lines = lines[: s - 1] + lines[e:]
        src = "\n".join(lines)
        # a class left without any body would not compile; ast.parse on the next loop catches it


if __name__ == "__main__":
    src = open(sys.argv[1], encoding="utf-8").read()
    out, removed = shake(src)
    compile(out, sys.argv[2], "exec")
    open(sys.argv[2], "w", encoding="utf-8").write(out)
    print("removed %d definitions, %d lines -> %d lines" % (len(removed), src.count("\n") - out.count("\n"),
                                                             out.count("\n")))
    for n, k in removed:
        print("  -", n, k)
