#!/usr/bin/env python3
"""Experiments 1-3: performance agreement (paper Eq 7),
error spread (Fig 1 analog), structural weighted Jaccard
(Eq 6) -- three arms: EZR-initial, EZR-refined (Table IV),
and branch (45 labels, best pruning of one tree).
Usage: python3 instability.py <dataset.csv>
Prints one CSV block; aggregate with summarize.py."""
import sys, random, itertools
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
from tools.ezr import (Data, csv, clone, adds, disty,
                       likely, Tree, treeLeaf, treeNodes,
                       the)
import branch as br

REPEATS = 20
SIGMAS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65,
          0.75, 0.85, 1]

def bused(w, lvl=0, names=None, cols=None):
  "branch tree -> {feature: weight}, paper Eq 6 weights."
  names = {} if names is None else names
  if w.at is not None:
    if lvl > 0:
      k = cols[w.at]
      names[k] = names.get(k, 0) + 1/(lvl+1)
    bused(w.yes, lvl+1, names, cols)
    bused(w.no,  lvl+1, names, cols)
  return names

def eused(data, tree):
  "EZR tree -> {feature: weight} (paper's own code)."
  names = {}
  for lvl, node in treeNodes(tree):
    if lvl == 0: continue
    _op, at, _y = node.how
    k = data.cols.names[at]
    names[k] = names.get(k, 0) + 1/(lvl+1)
  return names

def wjaccard(a, b):
  keys = set(a) | set(b)
  if not keys: return 1.0
  lo = sum(min(a.get(k,0), b.get(k,0)) for k in keys)
  hi = sum(max(a.get(k,0), b.get(k,0)) for k in keys)
  return lo/hi

def meanj(sets):
  js = [wjaccard(sets[i], sets[j]) for i, j in
        itertools.combinations(range(len(sets)), 2)]
  return sum(js)/len(js)

file = sys.argv[1]
name = file.split("/")[-1][:-4]
all_data = Data(csv(file))
ys  = [disty(all_data, r) for r in all_data.rows]
b4  = adds(ys)
win = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
b4w = adds([win(k) for k in ys])

tests_size = min(100, int(len(all_data.rows) * 0.5))
test  = clone(all_data, all_data.rows[:tests_size])
train = clone(all_data, all_data.rows[tests_size:])

ARMS = {"initial": dict(Budget=20, acq="xploit", leaf=2,
                        Impurity="entropy"),
        "refined": dict(Budget=50, acq="near", leaf=3,
                        Impurity="gini")}

out = {}
for arm, cfg in ARMS.items():
  for k, v in cfg.items(): setattr(the, k, v)
  trees = []
  for seed in range(REPEATS):
    the.seed = seed
    random.seed(seed)
    labels = likely(train)
    trees.append(Tree(clone(train, labels)))
  agree = {s: 0 for s in SIGMAS}
  for row in test.rows:
    preds = adds([win(treeLeaf(t, row).mu)
                  for t in trees])
    for s in SIGMAS:
      if preds.sd < s * b4w.sd: agree[s] += 1
  errs = []
  for t in trees:
    pick = min(test.rows,
               key=lambda r: treeLeaf(t, r).mu)
    best = min(test.rows, key=lambda r:
               disty(all_data, r))
    errs.append(win(disty(all_data, best))
                - win(disty(all_data, pick)))
  sets = [eused(train, t) for t in trees]
  out[arm] = (agree, adds(errs).sd, meanj(sets))

# branch arm: same split, branch's own d2h scale
names = list(all_data.cols.names)
btbl_all = br.Tbl(iter([names] +
                       [list(r) for r in all_data.rows]))
bys  = [br.disty(btbl_all, r) for r in btbl_all.rows]
bb4  = adds(bys)
bwin = lambda v: int(100*(1 - (v - bb4.lo)
                          / (bb4.mu - bb4.lo)))
bb4w = adds([bwin(k) for k in bys])
btrees, bfull = [], []
for seed in range(REPEATS):
  br.the.seed = seed
  random.seed(seed)
  btbl = br.Tbl(iter([names] + [list(r)
                for r in train.rows]))
  lab  = br.acquire(btbl, btbl.rows)
  full = br.Tree(btbl, lab)
  best = min(br.walk(full),
             key=lambda x: (x.score, x.leafs))
  btrees.append((btbl, best)); bfull.append(full)
agree = {s: 0 for s in SIGMAS}
for row in test.rows:
  preds = adds([bwin(bleaf := next(
      iter([br.leaf(tbl, t, list(row))])))
      for tbl, t in btrees])
  for s in SIGMAS:
    if preds.sd < s * bb4w.sd: agree[s] += 1
errs = []
for tbl, t in btrees:
  pick = min(test.rows,
             key=lambda r: br.leaf(tbl, t, list(r)))
  best = min(test.rows,
             key=lambda r: br.disty(btbl_all, r))
  errs.append(bwin(br.disty(btbl_all, best))
              - bwin(br.disty(btbl_all, pick)))
bsets  = [bused(t, 0, None, names) for _, t in btrees]
bfsets = [bused(t, 0, None, names) for t in bfull]
out["branch"] = (agree, adds(errs).sd, meanj(bsets))
out["branch_fulltree"] = (None, None, meanj(bfsets))

print(f"dataset, {name}")
print("sigmas, " + ", ".join(map(str, SIGMAS)))
for arm in ["initial", "refined", "branch"]:
  a, sd, mj = out[arm]
  row = [100*a[s]//tests_size for s in SIGMAS]
  print(f"{arm}_agreement, "
        + ", ".join(map(str, row)))
  print(f"{arm}_err_sd, {sd:.2f}")
  print(f"{arm}_jaccard, {mj:.3f}")
print(f"branch_fulltree_jaccard, "
      f"{out['branch_fulltree'][2]:.3f}")
