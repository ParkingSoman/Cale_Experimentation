"""P4 round 2: run several board.py rules in ONE torch process (one lock wait). Uses board.main unchanged.
Usage: r2_p4_multi.py --net clip --rules a,b --f .95,.85,.5 [any other board.py args]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board  # noqa: E402

args = sys.argv[1:]
i = args.index("--rules")
rules = args[i + 1].split(",")
rest = args[:i] + args[i + 2:]
for r in rules:
    sys.argv = ["board.py", "--rule", r] + rest
    try:
        board.main()
    except Exception as e:  # keep going; report
        import traceback
        traceback.print_exc()
        print(f"RULE FAILED {r}: {e}", flush=True)
