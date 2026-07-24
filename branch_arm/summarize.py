#!/usr/bin/env python3
"""Aggregate branch_arm/results/*.csv into one table:
per dataset, agreement@alpha=0.35, err_sd and weighted
Jaccard for the three arms, plus means and win/tie/loss
counts of branch vs refined."""
import glob, os, statistics as st

rows = []
here = os.path.dirname(os.path.abspath(__file__))
for fn in sorted(glob.glob(here + "/results/*.csv")):
    d = {}
    for line in open(fn):
        k, _, v = line.partition(",")
        d[k.strip()] = v.strip()
    if "branch_jaccard" not in d: continue
    ag = lambda arm: int(
        d[f"{arm}_agreement"].split(", ")[2])
    rows.append((d["dataset"],
        ag("initial"), ag("refined"), ag("branch"),
        float(d["initial_err_sd"]),
        float(d["refined_err_sd"]),
        float(d["branch_err_sd"]),
        float(d["initial_jaccard"]),
        float(d["refined_jaccard"]),
        float(d["branch_jaccard"]),
        float(d["branch_fulltree_jaccard"])))

W = max(len(r[0]) for r in rows)
print(f"{'dataset':{W}} | agree@.35 i/r/b | "
      "err_sd i/r/b | jaccard i/r/b/full")
for r in rows:
    print(f"{r[0]:{W}} | {r[1]:>3} {r[2]:>3} {r[3]:>3} | "
          f"{r[4]:>5.1f} {r[5]:>5.1f} {r[6]:>5.1f} | "
          f"{r[7]:.2f} {r[8]:.2f} {r[9]:.2f} {r[10]:.2f}")

col = lambda i: [r[i] for r in rows]
print(f"\n{'MEAN':{W}} | "
      f"{st.mean(col(1)):>3.0f} {st.mean(col(2)):>3.0f} "
      f"{st.mean(col(3)):>3.0f} | "
      f"{st.mean(col(4)):>5.1f} {st.mean(col(5)):>5.1f} "
      f"{st.mean(col(6)):>5.1f} | "
      f"{st.mean(col(7)):.2f} {st.mean(col(8)):.2f} "
      f"{st.mean(col(9)):.2f} {st.mean(col(10)):.2f}")
wtl = lambda i, j: (sum(1 for r in rows if r[i] > r[j]),
                    sum(1 for r in rows if r[i] == r[j]),
                    sum(1 for r in rows if r[i] < r[j]))
print("branch vs refined, agreement (w/t/l):",
      *wtl(3, 2))
print("branch vs refined, jaccard   (w/t/l):",
      *wtl(9, 8))
