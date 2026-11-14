"""Real dispatch preserves custom board identity and unknown attempts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


# The probe traces the transport client's opens with strace (Linux-only tool, Linux openat
# flags); without it the probe cannot prove the client never opens state.db writable.
@pytest.mark.platforms('linux')
@pytest.mark.skipif(shutil.which('strace') is None, reason='requires strace on PATH')
@pytest.mark.parametrize('mode', ['custom', 'cross_profile', 'restart'])
def test_owner_board_recovery(tmp_path, mode):
    from tests.gateway.fixtures.local_recovery_probe import seed_shared_uv_cache
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(tmp_path / 'user'), HERMES_HOME=str(tmp_path / 'state'),
               PYTHONPATH=str(repo), KANBAN_RECOVERY_MODE=mode)
    # This dispatches a kanban task to a worker that provisions its own pm-managed
    # venv; without a shared uv cache each of the 3 parametrised runs rebuilds a
    # fresh ~900MB venv into this disposable tmp_path (t_43e9dd1e).
    seed_shared_uv_cache(env['HERMES_HOME'])
    result = subprocess.run([sys.executable, str(repo / 'tests/gateway/fixtures/kanban_recovery_probe.py')],
                            cwd=repo, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    print(json.dumps(json.loads((tmp_path / 'state/receipt.json').read_text())))
