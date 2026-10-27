"""Render or submit one ACP job running a single evaluation command on a dedicated 8-GPU node.

Run from the repository root:
`python -m scripts.acp.submit --name <job> --log-dir <dir> [--submit] -- -m scripts.run_seed_tts ...`.
Arguments after `--` are passed to the repository Python on the node. Default only renders the request.
The job runs the live repository (it needs envs/, init_model/ and res/), so do not edit scripts it uses
while it runs.
"""

import argparse
import datetime
import json
import shlex
from pathlib import Path

import yaml

from .api import jobs_url, request

ROOT = Path(__file__).resolve().parents[2]


def payload(platform, name, log_dir, extra, spec=None):
    command = ["bash", str(ROOT / "scripts/acp/eval_job.sh"), str(ROOT), str(log_dir), *extra]
    return {
        "display_name": name,
        "framework": "PYTORCH",
        "roles": [
            {
                "name": "Worker",
                "resource_spec": [{"name": spec or platform["worker_spec"]}],
                "total_replicas": 1,
                "image_path": platform["image"],
                "startup_script": shlex.join(command),
            }
        ],
        "env": [
            {"key": "PYTHON_BIN", "value": platform["python"]},
        ],
        "mount": [
            {
                "type": "PV_AFS",
                "id": platform["volume_id"],
                "mount_path": platform["mount_path"],
                "subdir": "",
                "zone": platform["zone"],
            }
        ],
        "resource_pool": {"name": platform["cluster"], "zone": platform["zone"]},
        "scheduling": {"priority": "NORMAL", "quota_type": "RESERVED"},
        "ssh": {"auto_key_setup": platform["ssh_auto_key_setup"]},
        "fault_tolerance": {"backoff_limit": 0},
        "root_mapping": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="ACP display name")
    parser.add_argument("--log-dir", type=Path, required=True, help="Node log and submission records")
    parser.add_argument("--platform", type=Path, default=ROOT / "scripts/acp/acp.yaml")
    parser.add_argument("--spec", help="Resource spec instead of worker_spec")
    parser.add_argument("--submit", action="store_true", help="Create the job; else only render")
    args, extra = parser.parse_known_args()
    extra = [x for x in extra if x != "--"]
    if not extra:
        parser.error("pass the evaluation command after --")
    platform = yaml.safe_load(args.platform.read_text())
    log_dir = args.log_dir.resolve()
    if not log_dir.is_relative_to(platform["mount_path"]):
        parser.error("log-dir must be on the mounted ACP volume")
    body = payload(platform, args.name, log_dir, extra, args.spec)
    print(json.dumps({"submit": args.submit, "url": jobs_url(platform), "body": body}, indent=2))
    if not args.submit:
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submission = log_dir / "submissions" / stamp
    submission.mkdir(parents=True)
    (submission / "request.json").write_text(json.dumps(body, indent=2) + "\n")
    # Never retry POST: a lost response can still mean a job was allocated.
    result = request("POST", jobs_url(platform), body, home=platform["home"])
    (submission / "job.json").write_text(json.dumps({"submitted_at": stamp, "response": result}, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
