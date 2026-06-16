"""Run solve(ctx) on ONE local task with full visibility: prints every model code block and its
result, so we can see exactly where the loop goes wrong. Local + free. Usage:

  python debug_one.py 50e1ac9_1
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flywheel.appworld_env import AppWorldEnv  # noqa: E402
from flywheel.ctx import Ctx  # noqa: E402
import agent  # noqa: E402

PROXY_URL = os.environ.get("FLYWHEEL_URL", "https://homodeus-flywheel.fly.dev") + "/v1"


def main():
    tid = sys.argv[1] if len(sys.argv) > 1 else "50e1ac9_1"
    key = os.environ.get("FLYWHEEL_KEY", "")
    env = AppWorldEnv(tid, experiment_name="debug_one")
    ctx = Ctx(instruction=env.instruction, proxy_url=PROXY_URL, key=key,
              memory_dir=os.environ.get("FLYWHEEL_MEMORY_DIR", "./.memory"), env=env)

    print("=" * 80)
    print("TASK", tid, "::", env.instruction)
    print("=" * 80)

    orig_run = ctx.run_code
    orig_model = ctx.model
    state = {"turn": 0}

    def run_code(code):
        state["turn"] += 1
        print(f"\n----- run_code turn {state['turn']} -----\n{code}\n----- result -----")
        out = orig_run(code)
        print(str(out)[:2000])
        return out

    def model(messages, **kw):
        return orig_model(messages, **kw)

    ctx.run_code = run_code
    ctx.model = model

    agent.solve(ctx)

    verdict = env.world.evaluate().to_dict()
    print("\n" + "=" * 80)
    print("SUCCESS =", verdict.get("success"))
    print("verdict:", {k: verdict.get(k) for k in ("success", "difficulty", "num_tests")})
    for f in verdict.get("failures", []):
        print("FAIL:", f.get("label"), "-", f.get("requirement"))
    env.close()


if __name__ == "__main__":
    main()
