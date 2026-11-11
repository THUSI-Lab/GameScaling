"""UTC+8 for the eval tooling's wall-clock output (Python side of lib_tz.sh)."""
import os
import time


def use_utc8() -> None:
    """Pin this process to UTC+8 unless the caller already chose a zone.

    Keeps the submitters' progress lines on the same clock as the pod-side
    timestamps that lib_tz.sh fixes, so a submit log and the run directory it
    produced can be lined up even when the tooling runs on a UTC box. The
    zoneinfo database may be missing, hence the POSIX fallback -- its sign is
    inverted: "CST-8" means 8h EAST of UTC.
    """
    if os.environ.get("TZ"):
        return
    os.environ["TZ"] = ("Asia/Shanghai" if os.path.exists("/usr/share/zoneinfo/Asia/Shanghai")
                        else "CST-8")
    if hasattr(time, "tzset"):
        time.tzset()
