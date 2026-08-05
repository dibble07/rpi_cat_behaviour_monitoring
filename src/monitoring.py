import logging
import subprocess
import time

import psutil

from config import SYSTEM, settings
from shared import frame_queue, shutdown_event

logger = logging.getLogger(__name__)

# vcgencmd get_throttled bitmask — current-state bits (0–3) only
_THROTTLE_BITS = {
    0x1: "under-voltage",
    0x2: "freq-capped",
    0x4: "throttled",
    0x8: "soft-temp-limit",
}


def _get_throttle_flags() -> list[str]:
    """Get active throttle flags"""
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1
        ).stdout
        mask = int(out.strip().split("=")[1], 16)
        return [label for bit, label in _THROTTLE_BITS.items() if mask & bit]
    except Exception as exc:
        logger.warning(f"Throttle check unavailable: {exc}")
        return ["unknown"]


def monitoring_thread() -> None:
    """Monitor system resources and frame queue status."""
    logger.info("Monitoring thread started")
    # prepare psutil and initialise previous values
    process = psutil.Process()
    psutil.cpu_percent(percpu=True)
    process.cpu_percent()
    prev_rss = prev_cpu = prev_freq_mhz = prev_temp_c = prev_q_len = 0
    prev_disk_write_bytes = psutil.disk_io_counters().write_bytes
    last_mono = time.monotonic()

    while not shutdown_event.is_set():

        # timestamps and duration
        elapsed = time.monotonic() - last_mono
        last_mono = time.monotonic()

        # memory usage
        rss = process.memory_info().rss / (1024 * 1024)
        rss_delta = rss - prev_rss if prev_rss else 0.0
        prev_rss = rss

        # cpu utilization
        cpu = process.cpu_percent(interval=None)
        cpu_delta = cpu - prev_cpu if prev_cpu else 0.0
        prev_cpu = cpu

        # cpu frequency
        if SYSTEM == "Linux":
            freq_mhz = psutil.cpu_freq().current
            freq_delta = freq_mhz - prev_freq_mhz if prev_freq_mhz else 0.0
            prev_freq_mhz = freq_mhz

        # cpu temperature
        if SYSTEM == "Linux":
            temp_c = psutil.sensors_temperatures()["cpu_thermal"][0].current
            temp_delta = temp_c - prev_temp_c if prev_temp_c else 0.0
            prev_temp_c = temp_c

        # queue size
        q_len = frame_queue.qsize()
        q_delta = q_len - prev_q_len
        prev_q_len = q_len

        # disk write rate
        disk_bytes = psutil.disk_io_counters().write_bytes
        disk_mb_s = (disk_bytes - prev_disk_write_bytes) / (1024 * 1024) / elapsed
        prev_disk_write_bytes = disk_bytes

        # throttle state
        throttle_flags = _get_throttle_flags() if SYSTEM == "Linux" else []

        # nominality
        non_nominal = (
            rss >= 0.75 * 2 * 1024
            or cpu >= 0.75 * 4 * 100
            or (SYSTEM == "Linux" and temp_c >= 70)
            or q_len >= 5
            or disk_mb_s >= 50
            or throttle_flags
        )

        # log all metrics
        log = logger.warning if non_nominal else logger.info
        log(f"monitoring_loop_ms: {(time.monotonic() - last_mono) * 1000:.0f}")
        log(f"memory_mb: {rss:.0f} ({rss_delta:+.0f})")
        log(f"cpu_utilisation_pct: {cpu:.0f} ({cpu_delta:+.0f})")
        log(f"queue_size: {q_len} ({q_delta:+d})")
        log(f"disk_write_mbps: {disk_mb_s:.0f}")
        if SYSTEM == "Linux":
            log(f"cpu_freq_mhz: {freq_mhz:.0f} ({freq_delta:+.0f})")
            log(f"cpu_temp_c: {temp_c:.0f} ({temp_delta:+.0f})")
            log(
                f"throttle_state: {'|'.join(throttle_flags) if throttle_flags else 'ok'}"
            )

        time.sleep(0.5 if non_nominal else settings.MONITORING_PERIOD)

    logger.info("Monitoring thread stopped")
