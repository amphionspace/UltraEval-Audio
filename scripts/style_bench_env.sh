# Offline environment of the metrics/dots workers (same as run_seed_tts.offline_environment).
eval "$(envs/metrics/bin/python -c "
import shlex,sys;sys.path.insert(0,'scripts');from run_seed_tts import offline_environment
for k,v in offline_environment().items(): print('export '+k+'='+shlex.quote(v))")"
