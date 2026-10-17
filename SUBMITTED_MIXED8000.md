# TFI mixed8000 submitted version

This branch contains the exact source, model and ZIP used for official submission 3,
`tfi_mixed8000_extra.zip`, uploaded on 17 October 2026. The runnable `agent.py` and
all five supporting files at the repository root match the ZIP byte for byte.

## Official public result

| Opponent / metric | Result |
| --- | ---: |
| F, 1,000 games | 0.694 +/- 0.029 |
| O, 1,000 games | 0.538 +/- 0.031 |
| Public mean | 0.616 |
| Mean / worst agent CPU | 484 / 1,453 ms |
| Over-clocks / faults | 0 / 0 |

Intervals are the portal's 95% intervals. Snapshot: 17 October 2026, 15:28:33 IST.
O was evaluated through the official portal only.

## Model and search

The 53-feature, 32-hidden-unit MLP was trained on final-outcome labels from 8,000
games: 4,000 v3 versus F and 4,000 v3 versus v2. The dataset has 517,598 positions.
The outer split holds out 1,600 entire games; held-out accuracy is 76.38% and
log loss is 0.4668. Runtime scoring combines F's heuristic with the learned logit.

The shipped configuration is:

```json
{"top_k":4,"model":"model_mixed_extra.json","model_alpha":1000000.0,"open_m":6}
```

Search defaults retain four additional own actions per rollout, a 20 ms normal
search cap and a 150 ms placement search cap. These are soft deadlines, and the
agent also adapts to `obs.cpu_left`.

## Artifact verification

ZIP: [`submissions/tfi_mixed8000_extra.zip`](submissions/tfi_mixed8000_extra.zip)

SHA-256:

```text
9b0b95dd6eeda3388e5cfbb74dfbd69bfebc303c190564049bd6b4ae3e4793ca
```

Run the agent through the organisers' pinned engine and official kit. From the kit
root, use `python -m catan_kit.evaluate --a /path/to/hyde/agent.py --b examples/f_agent.py --games 1000`.

## Provenance

The search source was frozen from commit `31bda8f`. This branch starts from that
snapshot and adds the submitted evaluator, configuration, trained model and ZIP.
The submitted ZIP itself is the authoritative artifact.

Training dataset SHA-256:

```text
2898e20b95098b39f3859c7360e0ff1f25b48dfad283bbac0ca60cce6369b577
```

Public materials: the [organisers' kit](https://github.com/lyric-hyde/catan-hackathon-agent),
[catanatron](https://github.com/bcollazo/catanatron), scikit-learn and NumPy.
