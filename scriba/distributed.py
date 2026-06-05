from __future__ import annotations

import os


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def is_main_process() -> bool:
    rank = _env_int("RANK")
    if rank is not None:
        return rank == 0

    slurm = _env_int("SLURM_PROCID")
    if slurm is not None:
        return slurm == 0

    local_rank = _env_int("LOCAL_RANK")
    if local_rank is not None:
        return local_rank == 0

    return True

