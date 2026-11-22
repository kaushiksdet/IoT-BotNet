"""Run the whole experiment: python scripts/run_all.py [--data FILE] [--quick]"""
import subprocess, sys, os, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/UNSW_2018_IoT_Botnet_Dataset_2.csv"); ap.add_argument("--out", default="outputs"); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
here = os.path.dirname(__file__); q = ["--quick"] if a.quick else []
for cmd in [["01_review1_classical.py", "--data", a.data, "--out", a.out] + q, ["02_review2_deep.py", "--out", a.out] + q, ["03_stream_demo.py", "--out", a.out]]:
    print("\n=====", cmd[0], "====="); subprocess.run([sys.executable, os.path.join(here, cmd[0])] + cmd[1:], check=True)
