import logging
import os
import time

import psutil

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

        time.sleep(2)
