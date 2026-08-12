import logging
import os
import re
import subprocess
import time

import psutil

from config import SYSTEM, settings
from shared import (
    frame_queue,
    recording_queue_size,
    recording_raw_queue_size,
    shutdown_event,
)

logger = logging.getLogger(__name__)

_INT_MOUNT = "/"
_EXT_MOUNT = "/mnt/hdd"


# vcgencmd get_throttled bitmask — current-state bits (0–3) only
_THROTTLE_BITS = {
    0x1: "under-voltage",
    0x2: "freq-capped",
    0x4: "throttled",
    0x8: "soft-temp-limit",
}


def _base_device_name(device_path: str) -> str:
    """Convert partition device path to base block device name."""
    resolved = os.path.realpath(device_path)
    dev = resolved.split("/")[-1]
    if m := re.match(r"^(mmcblk\d+)p\d+$", dev):
        return m.group(1)
    if m := re.match(r"^(nvme\d+n\d+)p\d+$", dev):
        return m.group(1)
    return re.sub(r"\d+$", "", dev)


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

    # identify internal/external device names
    part = {p.mountpoint: p.device for p in psutil.disk_partitions()}
    _init_counters = psutil.disk_io_counters(perdisk=True)
    int_dev = _base_device_name(part[_INT_MOUNT]) if _INT_MOUNT in part else None
    int_dev = int_dev if int_dev and int_dev in _init_counters else None
    ext_dev = _base_device_name(part[_EXT_MOUNT]) if _EXT_MOUNT in part else None
    ext_dev = ext_dev if ext_dev and ext_dev in _init_counters else None
    if not int_dev or not ext_dev:
        logger.warning(
            f"Missing storage device: internal device = {int_dev}, external device = {ext_dev}"
        )

    # prepare psutil and initialise previous values
    process = psutil.Process()
    psutil.cpu_percent(percpu=True)
    process.cpu_percent()
    prev_rss = prev_cpu = prev_freq_mhz = prev_temp_c = 0
    prev_q_len = prev_recording_q_len = prev_raw_recording_q_len = 0
    prev_int_write_bytes = _init_counters[int_dev].write_bytes if int_dev else None
    prev_int_read_bytes = _init_counters[int_dev].read_bytes if int_dev else None
    prev_ext_write_bytes = _init_counters[ext_dev].write_bytes if ext_dev else None
    prev_ext_read_bytes = _init_counters[ext_dev].read_bytes if ext_dev else None
    swap = psutil.swap_memory()
    prev_swap_in_bytes = swap.sin
    prev_swap_out_bytes = swap.sout
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
        iowait_pct = float(
            getattr(psutil.cpu_times_percent(interval=None), "iowait", 0.0)
        )

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
        frame_q_len = frame_queue.qsize()
        frame_q_delta = frame_q_len - prev_q_len
        prev_q_len = frame_q_len
        recording_q_len = recording_queue_size
        recording_q_delta = recording_q_len - prev_recording_q_len
        prev_recording_q_len = recording_q_len
        raw_recording_q_len = recording_raw_queue_size
        raw_recording_q_delta = raw_recording_q_len - prev_raw_recording_q_len
        prev_raw_recording_q_len = raw_recording_q_len

        # disk write rate and free space
        _counters = psutil.disk_io_counters(perdisk=True)
        if int_dev:
            int_write_mb_s = (
                (_counters[int_dev].write_bytes - prev_int_write_bytes)
                / (1024**2)
                / elapsed
            )
            int_read_mb_s = (
                (_counters[int_dev].read_bytes - prev_int_read_bytes) / (1024**2)
            ) / elapsed
            prev_int_write_bytes = _counters[int_dev].write_bytes
            prev_int_read_bytes = _counters[int_dev].read_bytes
            int_free_gb = psutil.disk_usage(_INT_MOUNT).free / (1024**3)
        if ext_dev:
            ext_write_mb_s = (
                (_counters[ext_dev].write_bytes - prev_ext_write_bytes)
                / (1024**2)
                / elapsed
            )
            ext_read_mb_s = (
                (_counters[ext_dev].read_bytes - prev_ext_read_bytes) / (1024**2)
            ) / elapsed
            prev_ext_write_bytes = _counters[ext_dev].write_bytes
            prev_ext_read_bytes = _counters[ext_dev].read_bytes
            ext_free_gb = psutil.disk_usage(_EXT_MOUNT).free / (1024**3)

        # swap usage and rates
        swap = psutil.swap_memory()
        swap_used_mb = swap.used / (1024**2)
        swap_in_mb_s = ((swap.sin - prev_swap_in_bytes) / (1024**2)) / elapsed
        swap_out_mb_s = ((swap.sout - prev_swap_out_bytes) / (1024**2)) / elapsed
        prev_swap_in_bytes = swap.sin
        prev_swap_out_bytes = swap.sout

        # throttle state
        throttle_flags = _get_throttle_flags() if SYSTEM == "Linux" else []

        # nominality
        non_nominal = (
            rss >= 0.75 * 2 * 1024
            or cpu >= 0.75 * 4 * 100
            or iowait_pct >= 10
            or (SYSTEM == "Linux" and temp_c >= 70)
            or frame_q_len >= 5
            or recording_q_len >= 5
            or raw_recording_q_len >= 5
            or (int_dev is not None and int_write_mb_s >= 0.75 * 30)
            or (ext_dev is not None and ext_write_mb_s >= 0.75 * 70)
            or (int_dev is not None and int_free_gb < 1.0)
            or (ext_dev is not None and ext_free_gb < 5.0)
            or throttle_flags
        )

        # log all metrics in a single message to reduce logging overhead
        lines = [
            f"memory_mb: {rss:.0f} ({rss_delta:+.0f})",
            f"cpu_utilisation_pct: {cpu:.0f} ({cpu_delta:+.0f})",
            f"cpu_iowait_pct: {iowait_pct:.1f}",
            f"swap_used_mb: {swap_used_mb:.0f}",
            f"swap_in_mbps: {swap_in_mb_s:.1f}",
            f"swap_out_mbps: {swap_out_mb_s:.1f}",
            f"frame_queue_size: {frame_q_len} ({frame_q_delta:+d})",
            f"recording_queue_size: {recording_q_len} ({recording_q_delta:+d})",
            f"raw_recording_queue_size: {raw_recording_q_len} ({raw_recording_q_delta:+d})",
        ]
        if int_dev:
            lines.append(f"int_write_mbps: {int_write_mb_s:.0f}")
            lines.append(f"int_read_mbps: {int_read_mb_s:.0f}")
            lines.append(f"int_free_gb: {int_free_gb:.1f}")
        if ext_dev:
            lines.append(f"ext_write_mbps: {ext_write_mb_s:.0f}")
            lines.append(f"ext_read_mbps: {ext_read_mb_s:.0f}")
            lines.append(f"ext_free_gb: {ext_free_gb:.1f}")
        if SYSTEM == "Linux":
            lines.append(f"cpu_freq_mhz: {freq_mhz:.0f} ({freq_delta:+.0f})")
            lines.append(f"cpu_temp_c: {temp_c:.0f} ({temp_delta:+.0f})")
            lines.append(
                f"throttle_state: {'|'.join(throttle_flags) if throttle_flags else 'ok'}"
            )
        lines.append(f"monitoring_loop_ms: {(time.monotonic() - last_mono) * 1000:.0f}")

        log = logger.warning if non_nominal else logger.info
        log(" | ".join(lines))

        time.sleep(0.5 if non_nominal else settings.MONITORING_PERIOD)

    logger.info("Monitoring thread stopped")
