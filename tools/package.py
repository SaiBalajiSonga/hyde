"""Build a submission zip: agent.py and its data files at the top level, nothing from tools/.

    python tools/package.py v1            -> submissions/tfi_v1.zip
    python tools/package.py v2 --config tools/cfg/x.json   (shipped as config.json)

Then checks the zip: it extracts it to a temp dir, loads the Agent the way the harness does and
plays one decision-free import test.
"""
import argparse
import os
import shutil
import sys
import tempfile
import zipfile

TFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIP = ["agent.py", "value.py", "feats.py", "learned.py"]          # add more modules / data files here as the agent grows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version")
    p.add_argument("--config", default=None, help="json to ship as config.json")
    a = p.parse_args()

    out_dir = os.path.join(TFI, "submissions")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"tfi_{a.version}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in SHIP:
            z.write(os.path.join(TFI, f), f)
        if a.config:
            z.write(a.config, "config.json")
            import json
            cfgj = json.load(open(a.config))
            for key in ("model", "judge_model", "open_model"):
                if cfgj.get(key):
                    z.write(os.path.join(TFI, cfgj[key]), cfgj[key])
        elif os.path.exists(os.path.join(TFI, "config.json")):
            z.write(os.path.join(TFI, "config.json"), "config.json")
    names = zipfile.ZipFile(out).namelist()
    bad = [n for n in names if n.endswith((".so", ".pyd", ".dll", ".exe", ".dylib"))]
    assert not bad, bad

    tmp = tempfile.mkdtemp()
    try:
        zipfile.ZipFile(out).extractall(tmp)
        sys.path.insert(0, tmp)
        os.chdir(tmp)
        import importlib.util
        spec = importlib.util.spec_from_file_location("submitted_agent", os.path.join(tmp, "agent.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        agent = m.Agent()
        print("loaded OK; cfg:", {k: v for k, v in agent.cfg.items() if k != "weights"})
        print("weights:", agent.cfg["weights"])
    finally:
        os.chdir(TFI)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{out}  {os.path.getsize(out)} bytes  files={names}")


if __name__ == "__main__":
    main()
