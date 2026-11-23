# Vccm evaluation

The prepared subset contains 2,606 / 2,686 original rows: test 1,432 / 1,500,
undomain_style 91 / 100, undomain_timbre 1,083 / 1,086.
See raw_data/vccm_v1/prepare.json for missing audio by source. This is not a full
original-benchmark result. Speaker prompts are deterministic, different utterances
from the same speaker; emotion corpora use neutral prompts.

Scoring uses vendored ControlSpeech emotion classifier/checkpoint and pitch/energy
functions. A local API adapter maps old positional librosa mel arguments to keywords;
no thresholds or pitch/energy formulas are changed. Emotion2vec base v2.0.4 supplies
frame features. WER uses Whisper large-v3 and its English normalizer with corpus-level
word edit distance. Whisper runs on GPU in float16 (upstream script uses CPU), so exact
transcriptions need not be bit-identical. SIM uses WavLM base-plus-sv normalized x-vector
cosine against the speaker prompt. Speed accuracy requires MFA and is not computed.
`pitch_acc_official` counts a pitch success only when energy is also correct, matching
upstream; `pitch_acc` additionally reports independent pitch accuracy.

The ControlSpeech checkout is a local dependency excluded from Git. From the
repository root, provision it at the revision used for these results:

```bash
git clone https://github.com/jishengpeng/ControlSpeech.git third_party/controlspeech_eval
git -C third_party/controlspeech_eval checkout --detach 10ab00b4ce14989abeae0c475449629d3e173506
```

The audio corpus and local model files referenced by the preparation and scoring
scripts are also required.

From the repository root:

```bash
source scripts/style_bench_env.sh
CUDA_VISIBLE_DEVICES=6 envs/metrics/bin/python scripts/vccm_score.py \
  --reference --run-dir raw_data/vccm_v1/reference --limit 3
envs/metrics/bin/python scripts/vccm_score.py \
  --reference --summary --run-dir raw_data/vccm_v1/reference
CUDA_VISIBLE_DEVICES=6 envs/metrics/bin/python -u scripts/vccm_finish.py
```

The finish script scores reference recordings, waits for both CV3 and v5 generation
journals to contain every expected key, then scores each model and writes its summary.
It records progress/failure in res/vccm_logs/finish-status.json. Missing generated audio
is reported as failed and excluded from successful-row metric denominators. Check
failure counts alongside accuracy; comparisons should use the same successful keys
if either model has failures. A partial summary is not a final benchmark result.

Scoring shards use SHA256 of the item key so growing inference journals cannot shift
assignment. Keep num-shards constant during a run. Scores resume by key; do not reuse
score files after changing audio, models, or metric settings. Reference scores and
summaries are isolated from generated scores even when stored in one directory.
