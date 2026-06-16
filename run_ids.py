"""Run solve(ctx) on specific task ids (local, free). Prints PASS/FAIL + oracle reason on fail.

  python run_ids.py b119b1f_1 530b157_2 4ec8de5_1
"""
import json
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flywheel.appworld_env import AppWorldEnv  # noqa: E402
from flywheel.ctx import Ctx  # noqa: E402
from agent import solve  # noqa: E402

PROXY_URL = os.environ.get("FLYWHEEL_URL", "https://homodeus-flywheel.fly.dev") + "/v1"


def main():
    ids = sys.argv[1:] or ["b119b1f_1"]
    key = os.environ.get("FLYWHEEL_KEY", "")
    mem = tempfile.mkdtemp(prefix="fw_mem_")
    npass = 0
    for tid in ids:
        env = AppWorldEnv(tid, experiment_name="run_ids")
        ctx = Ctx(instruction=env.instruction, proxy_url=PROXY_URL, key=key, memory_dir=mem, env=env)
        try:
            solve(ctx)
        except Exception:
            traceback.print_exc()
        v = env.world.evaluate().to_dict()
        ok = bool(v.get("success"))
        npass += ok
        print(f"  {tid:14s} {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails = [f.get("trace", "").split("----------")[-1].strip()[:160]
                     for f in v.get("failures", [])]
            print("     reasons:", " | ".join(f for f in fails if f)[:300])
        env.close()
    print(f"\n{npass}/{len(ids)} pass")


if __name__ == "__main__":
    main()
