# branch arm: a third treatment for the instability study

Adds **branch** (vendored `branch.py`, an active-learning
tree pruner: 45 labels, one maxd-4 regression tree on
d2h, best pruning by (score, leafs)) as a third arm beside
the paper's EZR-initial and EZR-refined (Table IV)
configurations, under the paper's own measures:

- performance agreement (Eq 7): fixed test rows through
  all 20 models; agree when sd(models) < alpha * sd(data);
  alpha swept 0.15..1, headline 0.35
- error spread (Fig 1 analog): sd over 20 repeats of
  win(best test row) - win(recommended row)
- structural stability (Eq 6): root-weighted Jaccard over
  the 20 trees' feature sets; for branch, both the winner
  pruning and the full tree are reported

## Run

    python3 branch_arm/instability.py \
        data/optimize/misc/auto93.csv     # one dataset
    python3 branch_arm/summarize.py       # aggregate

`results/` holds outputs for 20 datasets spanning the
corpus (seconds per dataset; the whole 127 is an easy
sweep -- see below).

## Headline (20 datasets, alpha=0.35)

                    initial  refined  branch
    agreement          4%      28%     47%
    err_sd            19.2     20.2    19.2
    w. Jaccard        0.39     0.54    0.37  (full 0.42)

branch beats refined on agreement 13/5/2 and loses
structural similarity 19/20 -- simultaneously the most
performance-stable and least structurally stable arm.
One method, both of the paper's decoupled instabilities,
moving in opposite directions: supports RQ1.

## For the grad student

Goal: remove (or refute) the "needs more data" hedges on
RQ1/RQ3 of timm/src branch/REPORT-jul24.md. Both rigs run
on ALL 127 datasets, not just these 20.

**This rig (instability, all 127):**

1. Sweep: swap the 20-dataset loop for
   `find data/optimize -name '*.csv'` (seconds per
   dataset; an afternoon total). Outputs to build:
   Fig 6 analog (agreement counts across the alpha
   sweep, initial/refined/branch lines), Fig 7 analog
   (weighted Jaccard per dataset), Fig 1 analog
   (error spread per dataset).
2. Gate with the paper's stats (tools/stats.py: KS 95% +
   Cliff's <= 0.195 + top-set ranking), not raw means --
   deliverable is "branch statistically top-ranked on k
   of 127", the paper's Fig 8/9 currency.
3. Add the clusterer arms (RQ3 code: KMeans, HDBSCAN,
   CURE -- the paper's stability champions) to the same
   table. Only then is any "high water mark" claim
   testable.

**Sister rig (JSS'26 feature selection, also all 127):**
extend `experiments/branch_fs/` (PR 1 on
amiiralii/Minimal-Data-Maximum-Clarity) from its 20 to
all 127. The JSS repo ships only its 60 datasets; this
clone's data/optimize/ has all 127 -- point the FS rig
here. Report the 60-subset separately (direct comparison
against the published Table 10 / Figs 11-12), full-127
as headline. Cost warning: the SVR (O(n^2)) and torch-ANN
arms take hours EACH on the 100k-row / 1000-column
monsters -- use a server, or drop those two arms on the
extension datasets (lgbm is the headline column; say so
in threats). The 127-run's biggest payoff: Fig 11/12's
100+ feature bin grows from 2 datasets to the whole
FM/FFM/Scrum family, so the wide-data verdict on branch
stops resting on FFM-125 alone. Also worth one extra
sweep there: budget scaled with column count (is the
FFM-125 miss just label starvation? SHAP at the same N=4
scores 92).

**Cheap fix candidate (either rig):** branch's
winner-pruning Jaccard (0.37) < its full-tree Jaccard
(0.42): the (score, leafs) tiebreak adds structural
churn. A stability-aware tiebreak (prefer the previous
run's shape among score-ties) cannot hurt performance.
