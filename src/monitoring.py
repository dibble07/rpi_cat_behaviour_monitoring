import faulthandler
import logging
import os
import time

import psutil

from config import SYSTEM, settings
from shared import frame_queue, get_heartbeat_snapshot, shutdown_event

logger = logging.getLogger(__name__)
process = psutil.Process(os.getpid())


def monitoring_thread() -> None:
    """Monitor system resources and frame queue status."""
    # prepare percent sampling
    psutil.cpu_percent(percpu=True)
    process.cpu_percent()
    prev_q_len = 0
    prev_rss = 0.0
    prev_disk_write_bytes: int | None = None
    stale_processing_intervals = 0

    while not shutdown_event.is_set():

        # memory usage
        rss = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory: {rss:.0f} MB")
        rss_delta = rss - prev_rss if prev_rss else 0.0
        prev_rss = rss

        # CPU usage
        proc_cpu = process.cpu_percent(interval=None)
        logger.info(f"ProcCPU: {proc_cpu:.1f}%")

        # thread counts
        num_threads = process.num_threads()
        logger.debug(f"Threads: {num_threads}")

        # queue size
        q_len = frame_queue.qsize()
        logger.info(f"Frame queue length: {q_len}")
        q_delta = q_len - prev_q_len
        prev_q_len = q_len

        # processing/writer heartbeat ages
        (
            processing_hb,
            writer_hb,
            processing_evt,
            writer_evt,
        ) = get_heartbeat_snapshot()
        now_mono = time.monotonic()
        processing_age = now_mono - processing_hb if processing_hb else float("inf")
        writer_age = now_mono - writer_hb if writer_hb else float("inf")
        logger.info(
            "Heartbeat age [s] - processing: %.1f (%s), writer: %.1f (%s)",
            processing_age,
            processing_evt,
            writer_age,
            writer_evt,
        )
        hb_warn_age = 2 * settings.MONITORING_PERIOD
        if processing_age > hb_warn_age:
            stale_processing_intervals += 1
            logger.warning(
                "Processing heartbeat stale: %.1f s since last event (%s)",
                processing_age,
                processing_evt,
            )
            if stale_processing_intervals >= settings.MONITORING_STALE_DUMP_INTERVALS:
                logger.warning(
                    "Processing stale for %s intervals - dumping all thread stacks",
                    stale_processing_intervals,
                )
                faulthandler.dump_traceback(all_threads=True)
        else:
            stale_processing_intervals = 0
        if writer_hb and writer_age > hb_warn_age:
            logger.warning(
                "Writer heartbeat stale: %.1f s since last event (%s)",
                writer_age,
                writer_evt,
            )

        # log interval deltas for trend diagnostics
        logger.info(
            "Monitor deltas: rss_delta_mb=%.1f frame_queue_delta=%s",
            rss_delta,
            q_delta,
        )
        if q_delta >= settings.MONITORING_QUEUE_GROWTH_WARN:
            logger.warning(
                "Frame queue grew by %s in one interval (current=%s)",
                q_delta,
                q_len,
            )
        if rss_delta >= settings.MONITORING_RSS_GROWTH_WARN_MB:
            logger.warning(
                "Memory grew by %.1f MB in one interval (current=%.0f MB)",
                rss_delta,
                rss,
            )

        # disk I/O wait
        disk_io = psutil.disk_io_counters()
        disk_write_mb = disk_io.write_bytes / (1024 * 1024)
        if prev_disk_write_bytes is None:
            disk_write_mb_s = 0.0
        else:
            disk_write_mb_s = (
                (disk_io.write_bytes - prev_disk_write_bytes)
                / (1024 * 1024)
                / settings.MONITORING_PERIOD
            )
        prev_disk_write_bytes = disk_io.write_bytes
        logger.info("Disk write throughput: %.1f MB/s", disk_write_mb_s)
        if disk_write_mb_s >= settings.MONITORING_DISK_WRITE_WARN_MB_S:
            logger.warning(
                "Disk write throughput high: %.1f MB/s",
                disk_write_mb_s,
            )
        match SYSTEM:
            case "Linux":
                logger.debug(
                    f"Disk write: {disk_write_mb:.0f} MB total | busy_time: {disk_io.busy_time} ms"
                )
            case "Darwin":
                logger.debug(f"Disk write: {disk_write_mb:.0f} MB total")
            case _:
                logger.debug(f"Disk write: {disk_write_mb:.0f} MB total")

        time.sleep(settings.MONITORING_PERIOD)
