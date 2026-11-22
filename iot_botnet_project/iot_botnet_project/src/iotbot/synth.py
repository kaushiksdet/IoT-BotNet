"""Synthetic Bot-IoT-style generator. Writes the SAME 35-column headerless CSV as UNSW Bot-IoT,
so every script runs unchanged on synthetic or real data. Class mix and structure are illustrative only."""
import numpy as np, pandas as pd
from .common import BOT_IOT_COLUMNS

def make_synthetic(n=60_000, seed=7):
    r = np.random.default_rng(seed)
    cls = r.choice(["Normal", "Reconnaissance", "DoS", "Theft"], size=n, p=[0.010, 0.680, 0.300, 0.010])
    rows = []
    for i, c in enumerate(cls):
        if c == "Normal":
            sub = "Normal"; proto = r.choice(["tcp", "udp", "icmp"], p=[.55, .40, .05])
            dport = int(r.choice([53, 80, 443, 123, 1883, int(r.integers(1024, 65535))], p=[.45, .12, .15, .05, .05, .18]))
            dur = r.lognormal(.2, 1.6); pk = 1 + r.lognormal(2.0, 1.3); by = pk * r.lognormal(5.0, .9)
            rate = pk / (dur + .01) * r.lognormal(0, .5); state = r.choice(["CON", "FIN", "REQ", "RST"], p=[.55, .40, .03, .02]); ts = r.uniform(0, 1)
        elif c == "Reconnaissance":
            sub = r.choice(["OS_Fingerprint", "Service_Scan"], p=[.43, .57]); proto = "tcp" if r.random() < .92 else "udp"
            dport = int(r.choice([22, 23, 80, 443, 8080, 445, 3389, int(r.integers(1, 65535))], p=[.12, .1, .2, .15, .1, .1, .08, .15])) if r.random() > .01 else 53
            dur = r.lognormal(-1.2, .5); pk = 1 + r.poisson(2.2); by = pk * r.lognormal(4.2, .5)
            rate = pk / (dur + .01) * r.lognormal(0, .35); state = r.choice(["REQ", "RST", "CON"], p=[.58, .38, .04]); ts = np.clip(r.normal(.35, .12), 0, 1)
            if r.random() < .004: dur = r.lognormal(.2, 1.2); pk = 1 + r.lognormal(1.8, 1.0); by = pk * r.lognormal(5, .8); rate = pk / (dur + .01); state = "CON"; dport = int(r.choice([80, 443, 53]))
        elif c == "DoS":
            sub = r.choice(["TCP", "HTTP"], p=[.82, .18]); proto = "tcp" if r.random() < .97 else "udp"
            dport = int(r.choice([80, 443, 8080, 1883, int(r.integers(1, 65535))], p=[.45, .15, .15, .1, .15]))
            dur = r.lognormal(.6, .5); pk = 10 + r.lognormal(4.3, .5); by = pk * r.lognormal(4.8, .4)
            rate = pk / (dur + .01) * r.lognormal(0, .3); state = r.choice(["REQ", "CON", "RST"], p=[.65, .05, .30]); ts = np.clip(r.normal(.65, .1), 0, 1)
            if r.random() < .003: dur = r.lognormal(.2, 1.2); pk = 1 + r.lognormal(1.9, 1.0); by = pk * r.lognormal(5, .8); rate = pk / (dur + .01); state = "FIN"
        else:
            sub = r.choice(["Keylogging", "Data_Exfiltration"], p=[.93, .07]); proto = "tcp"
            dport = int(r.choice([8080, 21, 22, 443, int(r.integers(1024, 65535))], p=[.35, .2, .1, .15, .2]))
            dur = r.lognormal(.9, .8); pk = 1 + r.lognormal(2.6, .5); by = pk * r.lognormal(5.6, .4)
            rate = pk / (dur + .01) * r.lognormal(0, .6); state = r.choice(["CON", "FIN", "REQ"], p=[.55, .3, .15]); ts = r.uniform(0, 1)
            if r.random() < .08: dport = int(r.choice([53, 80, 443])); proto = r.choice(["tcp", "udp"])
        sp = pk * r.uniform(.45, .75); st = 1.5e9 + ts * 1e5
        rows.append([i, st, "e", proto, f"192.168.100.{int(r.integers(1,250))}", int(r.integers(1024, 65535)),
                     f"192.168.100.{int(r.integers(1,250))}", dport, pk, by, state, st + dur, i, dur, by / pk,
                     by / pk * r.uniform(.05, .6), "00:00:00:00:00:00", "00:00:00:00:00:00", by, by / pk * .5, by / pk * 1.5,
                     "x", "x", "x", "x", sp, max(pk - sp, 0), by * r.uniform(.5, .8), by * r.uniform(.2, .5), rate, rate * .6, rate * .4,
                     int(c != "Normal"), c, sub])
    return pd.DataFrame(rows, columns=BOT_IOT_COLUMNS)
