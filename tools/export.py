"""Bake trained parameters into a standalone submission file.

usage: python tools/export.py tools/params_best.json [theta|champion] out.py
"""
import json, re, sys
src = open("bot.py").read()
params = json.load(open(sys.argv[1]))[sys.argv[2]]
start = src.index("PARAMS = {")
end = src.index("}\n", start) + 2
body = "PARAMS = {\n" + "".join(f"    {k!r}: {round(v, 4)!r},\n" for k, v in sorted(params.items())) + "}\n"
open(sys.argv[3], "w").write(src[:start] + body + src[end:])
print("wrote", sys.argv[3])
