import logging
import os
import time

import psutil

from config import SYSTEM
from shared import frame_queue, shutdown_event

logger = logging.getLogger(__name__)
process = psutil.Process(os.getpid())


def monitoring_thread():

    # prepare percent sampling
    psutil.cpu_percent(percpu=True)
    process.cpu_percent()

    while not shutdown_event.is_set():

        # memory usage
        rss = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory: {rss:.0f} MB")

        # CPU usage
        proc_cpu = process.cpu_percent(interval=None)
        logger.info(f"ProcCPU: {proc_cpu:.1f}%")

        # thread counts
        num_threads = process.num_threads()
        logger.info(f"Threads:{num_threads}")

        # queue size
        q_len = frame_queue.qsize()
        logger.info(f"Frame queue length: {q_len}")

        # disk I/O wait
        disk_io = psutil.disk_io_counters()
        disk_write_mb = disk_io.write_bytes / (1024 * 1024)
        match SYSTEM:
            case "Linux":
                logger.info(f"Disk write: {disk_write_mb:.0f} MB total | busy_time: {disk_io.busy_time} ms")
            case "Darwin":
                logger.info(f"Disk write: {disk_write_mb:.0f} MB total")
            case _:
                logger.info(f"Disk write: {disk_write_mb:.0f} MB total")

        time.sleep(2)
