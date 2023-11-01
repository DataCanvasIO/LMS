import asyncio
import re
import time
import traceback

import httpx as httpx
import psutil
from psutil import NoSuchProcess, AccessDenied
from sanic import Sanic
from sanic.response import json

from lms.client.config import server_id
from lms.client.daemon.nvml_util import nvml_pmon

LMS_RT_PATTERN = rf'.*(lms_rt\s+deploy|deepspeed_infer\.py|transformers_infer\.py).*--model_path=.*?\s+--model_name=(?P<model_name>.*?)\s+(--port=(?P<port>\d*))?\s*--server_id={server_id}'

metrics_map = {}
last_update_time = time.time_ns()


def make_app(interval):
    app = Sanic(name="default")

    async def collect_metrics(interval=15):
        while True:
            pids = psutil.pids()
            gpu_stats = nvml_pmon()
            timestamp = time.time()
            common_metrics_map = {}
            for pid in pids:
                try:
                    p = psutil.Process(pid)
                    m = re.match(LMS_RT_PATTERN, ' '.join(p.cmdline()))
                    if m:
                        model_name = m.group('model_name')
                        common_metrics = common_metrics_map.get(model_name, {})

                        common_metrics['cpu_percent'] = common_metrics.get('cpu_percent', 0) + p.cpu_percent(interval=1)
                        common_metrics['mem_percent'] = common_metrics.get('mem_percent', 0) + p.memory_percent()
                        memory_info = p.memory_info()
                        common_metrics['mem_usage'] = common_metrics.get('mem_usage', 0) + memory_info.rss
                        gpu_info = gpu_stats.get(str(pid), {})
                        common_metrics['gpu_usage_percent'] = common_metrics.get('gpu_usage_percent', 0) + float(
                            gpu_info.get("sm", 0) if gpu_info.get("sm", 0) != '-' else 0)
                        common_metrics['gpu_mem_usage_percent'] = common_metrics.get('gpu_mem_usage_percent',
                                                                                     0) + float(
                            gpu_info.get("mem", 0) if gpu_info.get("mem", 0) != '-' else 0)
                        common_metrics['gpu_mem_usage'] = common_metrics.get('gpu_mem_usage', 0) + float(
                            gpu_info.get("fb", 0) if gpu_info.get("fb", 0) != '-' else 0)
                        common_metrics['timestamp'] = timestamp
                        if len(gpu_info) > 0:
                            gpu_details = common_metrics.get('gpu_details', [])
                            gpu_details.append({
                                "device": gpu_info.get("gpu", "N/A"),
                                "gpu_usage_percent": gpu_info.get("sm", "N/A"),
                                "gpu_mem_usage_percent": gpu_info.get("mem", "N/A"),
                                "gpu_mem_usage": gpu_info.get("fb", "N/A")
                            })
                            common_metrics['gpu_details'] = gpu_details

                        port = m.group('port')
                        if port:
                            try:
                                proxies = {
                                    "all://127.0.0.1": None,
                                }
                                async with httpx.AsyncClient(proxies=proxies) as client:
                                    resp = await client.get(
                                        url="http://127.0.0.1:%s/metrics" % port
                                    )
                                if resp.status_code == 200:
                                    common_metrics['total_requests'] = resp.json()['total_requests']
                                else:
                                    print(resp.status_code)
                                    common_metrics['total_requests'] = 'N/A'
                            except Exception:
                                traceback.print_exc()
                                common_metrics['total_requests'] = 'N/A'

                        common_metrics_map[model_name] = common_metrics

                except NoSuchProcess:
                    pass
                except AccessDenied:
                    pass
            global metrics_map
            metrics_map = common_metrics_map
            global last_update_time
            last_update_time = time.time_ns()
            await asyncio.sleep(interval)

    app.add_task(collect_metrics(interval))

    @app.get("/metrics")
    async def metrics(request):
        _metrics = []
        for k, v in metrics_map.items():
            v['model_name'] = k
            if not v.__contains__('gpu_details'):
                v['gpu_details'] = []
            _metrics.append(v)
        return json({"timestamp": last_update_time, "metrics": _metrics})

    @app.get("/ready")
    async def ready(request):
        return json({})

    @app.after_server_start
    async def listener(app, loop):
        print("started")

    return app
