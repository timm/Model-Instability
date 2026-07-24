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

1. Sweep all 127: swap the 20-dataset loop for
   `find data/optimize -name '*.csv'`.
2. Gate claims with the paper's stats (KS + Cliff's +
   top-set), not raw means.
3. Add the clusterer arms (RQ3 code) to test the "high
   water mark" claim against the stablest baselines.
4. branch's winner-pruning Jaccard (0.37) < its full-tree
   Jaccard (0.42): the (score, leafs) tiebreak adds
   structural churn. A stability-aware tiebreak (prefer
   the previous run's shape among score-ties) is a cheap
   candidate fix that cannot hurt performance.
