# Round 2: what is run, in which order, and how to read it

Four training experiments (88 runs, about 11 h on the 6-GPU / 288-core machine) and four evaluation-only analyses (no training, about 1-2 h).
All use the same recipe as the first noise grid (10 M steps, 32 envs, rollout 16384, terrain 2-6 cm drawn per episode, Stage-1 policy frozen).

| Question | Experiment | Runs | Compared with (already exists) |
| --- | --- | --- | --- |
| How much preview is needed? | `horizon`: P2 with the first 1, 2, 3, 4 look-ahead points (0.2 ... 0.8 m), C1 and C2, level P | 32 | no preview (P1) and five points up to 1.0 m (P2), level P |
| Which imperfection hurts, and does memory help against it? | `factors`: noise only / delay only / dropout only, each at the size of level Q, P2, C1 and C2 | 24 | Q (all three together) and 0 (clean), P2 |
| Does the GRU setting matter? | `gru`: window K = 8, 32 (hidden 32) and hidden 16, 64 (K = 16), level P, P2, C2 | 16 | K = 16, hidden 32 (level P, P2) |
| Is the memory effect real? | replication seeds 4-7, levels P and Q, P2 | 16 | seeds 0-3, same tags |
| Does C2 use its history? | `scripts/gru_ablation.py` (evaluation only) | 0 | the real window |
| Reward, SD, error bars | re-evaluation of the existing runs into a new folder | 0 | - |
| Videos C1 vs C2 | `scripts/compare_videos.py` (median run of each group) | 0 | - |

## Order on UCloud

```bash
source /work/ai3_test_train/.ucloud_env.sh
cd /work/ai3_test_train/terrain-memory-prosthesis      # the repository, with this patch applied
export STAGE1=<the same Stage-1 session folder as for the first grid>

# 1. start the training (72 runs; PAR=20 leaves about 100 cores for the evaluation-only jobs below)
DRY=1 EXPS="horizon factors gru" bash scripts/ucloud/exp_grid.sh | head      # check the list first
PAR=20 EXPS="horizon factors gru" bash scripts/ucloud/exp_grid.sh
# 2. the 16 replication runs go into the existing level folders; start them when step 1 is done (or PAR=12 now, both queues then share the cores)
LEVELS="P Q" CONDS="P2" SEEDS="4 5 6 7" PAR=16 bash scripts/ucloud/noise_grid.sh

# 3. at once, while it trains: evaluation-only
OUT=/work/ai3_test_train/eval_noise_r CKPTS=1 WORKERS=96 LEVELS="0 P Q" bash scripts/ucloud/eval_noise.sh      # adds the reward columns, last checkpoint only
#    (run it again, same command, after the replication seeds 4-7 are finished: it then holds 8 seeds of P and Q; the old eval_noise folder stays untouched)
```

Progress: `ls ../myoassist/rl_train/results | grep -c -E '_(hz|f[NDR]|g[0-9])'` (72 when all runs exist), failures: `grep FAILED ../exp_logs/queue_*.log`.
After a node loss: run the same command again, only when `pgrep -f train_stage2` shows nothing (it continues the unfinished runs with their own noise).

## Evaluation-only analyses (can run at any time)

```bash
W=/work/ai3_test_train; WS=$W; R=$W/myoassist/rl_train/results
# does C2 use its history? (level P and Q, P2: damage the window and see what is lost)
python scripts/gru_ablation.py --runs $R/stage2_P2_gru_s*_nP_* --out $W/ablation_P --workers 96 --episodes 20
python scripts/gru_ablation.py --runs $R/stage2_P2_gru_s*_nQ_* --out $W/ablation_Q --workers 96 --episodes 20
# videos: median run of C1 and of C2, same terrain and episode seeds, left C1 | right C2
python scripts/compare_videos.py --left-csv $W/eval_noise_r/P_mlp/stage2_eval.csv --right-csv $W/eval_noise_r/P_gru/stage2_eval.csv \
    --cond P2 --results $R --out $W/videos_c1_c2_P --seeds 0,1,2 --amp 0.06
```
Reading the ablation: `repeat` removes all temporal information but stays a realistic input; `shuffle` keeps the values and destroys the order;
`zero` removes the window altogether. If the steps survived do not drop, the policy does not use the memory.

## After the training: evaluate and compare

```bash
OUT_E=$W/eval_exp; E=$W/eval_noise_r      # the re-evaluated folder: last checkpoint, with reward
bash scripts/ucloud/eval_exp.sh                      # -> $OUT_E/<tag>_<mlp|gru>/stage2_eval.csv
# preview horizon, C1 (repeat with mlp -> gru, "C2")
python scripts/compare_groups.py "none (P1)=$E/P_mlp/stage2_eval.csv@P1" "to 0.2 m=$OUT_E/hz1_mlp/stage2_eval.csv" \
   "to 0.4 m=$OUT_E/hz2_mlp/stage2_eval.csv" "to 0.6 m=$OUT_E/hz3_mlp/stage2_eval.csv" "to 0.8 m=$OUT_E/hz4_mlp/stage2_eval.csv" \
   "to 1.0 m=$E/P_mlp/stage2_eval.csv@P2" --ref "none (P1)" --out $W/horizon_C1 --title "C1: steps survived vs preview length (level P)"
# noise / delay / dropout one at a time (P2), C1 and C2 -> change mlp to gru for C2
python scripts/compare_groups.py "clean (0)=$E/0_mlp/stage2_eval.csv@P2" "noise only=$OUT_E/fN_mlp/stage2_eval.csv" \
   "delay only=$OUT_E/fD_mlp/stage2_eval.csv" "dropout only=$OUT_E/fR_mlp/stage2_eval.csv" "all three (Q)=$E/Q_mlp/stage2_eval.csv@P2" \
   --ref "clean (0)" --out $W/factors_C1
# GRU settings (C2, level P, P2)
python scripts/compare_groups.py "K16 h32=$E/P_gru/stage2_eval.csv@P2" "K8 h32=$OUT_E/g8_32_gru/stage2_eval.csv" "K32 h32=$OUT_E/g32_32_gru/stage2_eval.csv" \
   "K16 h16=$OUT_E/g16_16_gru/stage2_eval.csv" "K16 h64=$OUT_E/g16_64_gru/stage2_eval.csv" --ref "K16 h32" --out $W/gru_settings
# replication: the seed statistics with all seeds (after the second eval_noise_r run above)
python scripts/seed_stats.py 0/C1=$E/0_mlp/stage2_eval.csv 0/K16=$E/0_gru/stage2_eval.csv P/C1=$E/P_mlp/stage2_eval.csv P/K16=$E/P_gru/stage2_eval.csv \
   Q/C1=$E/Q_mlp/stage2_eval.csv Q/K16=$E/Q_gru/stage2_eval.csv --noisy P,Q --clean 0 --metric steps | tee $W/seed_stats_8seeds.txt
# reward
python scripts/compare_groups.py "C1 P2=$E/P_mlp/stage2_eval.csv@P2" "C2 P2=$W/eval_noise_r/P_gru/stage2_eval.csv@P2" --metric reward --out $W/reward_P
```

## What each result can and cannot say

* All of it is 4 seeds per cell (8 for the replicated levels): differences below about 50 steps are not detectable. A bar chart with SD and dots shows
  this directly; read the interval, not only p. Several groups are compared against one reference: multiply p by the number of groups (Bonferroni).
* `horizon`: the preview effect as a curve over the number of look-ahead points. The points are cumulative (0.2 m, then 0.2 + 0.4 m, ...); a far point alone
  is not tested. If the curve is flat after 0.4 m, a short preview is enough.
* `factors`: each imperfection at the size of Q; the interaction between them is not separable from this design.
* `gru`: K = 8 / 32 and hidden size are varied one at a time around K = 16, hidden 32, not as a full grid.
* The ablation says whether the memory is used, not whether it is useful in other conditions.
* The first report figure pooled the checkpoints from 7.9 M steps (`plot_noise.py`); `seed_stats.py` and `compare_groups.py` use only the last checkpoint
  (differences < 15 steps in the first grid). With the single-checkpoint folder `eval_noise_r` all tools give the same means.
