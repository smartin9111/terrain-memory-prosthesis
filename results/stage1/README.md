# Stage-1 results (final policy)

Policy: UCloud session `train_session_20261012-174642`, checkpoint `model_29999104.zip` (30M steps, config `stage1_flat_final.json`).

Put here the **small** files of `stage1_results.tgz` (CSV, JSON, PNG, logs of the checks 00, 02, 03a and the `eval_final.log`):

```
tar -xzf stage1_results.tgz -C results/stage1 --exclude='*.mp4' --exclude='*.zip' --exclude='*.npz'
```

Not in git: the checkpoint (attach `model_29999104.zip` + the session's `session_config.json` to a GitHub Release of the tag
`stage1-final`) and the videos (`walk_seed*.mp4`).
