import asyncio
import logging
import mimetypes
import traceback
from collections import OrderedDict
from urllib.parse import unquote

import httpx as httpx
from requests.exceptions import *
from sanic import Sanic, NotFound, file, response
from sanic.response import json

from lms.web.internal.route import make_models
from lms.web.utils import traceback_utils
from lms.web.utils.asyncssh_utils import write_file
from lms.web.utils.peewee_repo import *

logger = logging.getLogger(__name__)

mimetypes.add_type('text/markdown', '.md')
mimetypes.add_type('text/plain', '.sh')


def get_model_path(model_path):
    return model_path[model_path.index(":") + 1:]


def unique_combine(a, b):
    c = []
    for i in b:
        if i in a or i in c:
            pass
        else:
            c.append(i)
    a.extend(c)
    return a


available_map = {}


def make_app(server_id):
    app = Sanic(name="lms_web")

    import pathlib
    app.static('/static', str(pathlib.Path(__file__).parent.resolve()) + '/static', name="static")
    app.static("/", str(pathlib.Path(__file__).parent.resolve()) + '/static/index.html', name="index")
    app.static("/ModelList", str(pathlib.Path(__file__).parent.resolve()) + '/static/index.html', name="mlist")
    app.static("/ModelMonitoring", str(pathlib.Path(__file__).parent.resolve()) + '/static/index.html', name="mmonitor")
    app.static("/Guide", str(pathlib.Path(__file__).parent.resolve()) + '/static/index.html', name="guide")

    async def sync_task(app):
        while True:
            _available_map = {}
            for model in query_model_list(status='deployed'):
                try:
                    ready_url = model['api_url'].replace('/prediction', '/ready')
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(url=ready_url)

                    if resp.status_code == 200:
                        _available_map[model['model_name']] = True
                    else:
                        _available_map[model['model_name']] = False
                except Exception:
                    _available_map[model['model_name']] = False

            global available_map
            available_map = _available_map
            await asyncio.sleep(5)

    app.add_task(sync_task(app))

    blue_internal = make_models()
    app.blueprint(blue_internal)

    @app.get("/ModelList/<filename:path>")
    async def index1(request, filename):
        return await file(str(pathlib.Path(__file__).parent.resolve()) + '/static/index.html')

    @app.get("/ModelMonitoring/<filename:path>")
    async def index2(request, filename):
        return await file(str(pathlib.Path(__file__).parent.resolve()) + '/static/index.html')

    @app.route("/lms")
    async def test2(request):
        return json({"hello": "world"})

    @app.get("/lms/models")
    async def model_list(request):
        """
        查询 模型列表
        :param request:
        :return:
        """
        print(available_map)
        data = query_model_list()
        for i in data:
            if i['status'] == 'deployed':
                i['available'] = available_map.get(i['model_name'], True)
        result = {'models': data}
        return json(result)

    @app.get("/lms/models/<model_name>")
    async def model_detail(request, model_name):
        """
         查询 模型详情
        :param request:
        :param model_name:
        :return:
        """
        model_name = unquote(model_name)
        if model_name is None:
            raise Exception("model model_name cannot be empty !")

        m = query_model_by_model_name(model_name)

        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        return json(m)

    @app.get("/lms/models/<model_name>/ls")
    async def model_file_ls(request, model_name):
        """
         查询 模型文档结构
        :param request:
        :param model_name:
        :param <path> :
        :return:
        """
        args = request.args
        path = args.get('path', "/")
        model_name = unquote(model_name)
        m = query_model_by_model_name(model_name)
        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        model_path = m.get("model_path")

        if model_path is None:
            raise Exception("model model_path: [{}] is not exists ! ".format(model_name))

        from lms.web.utils.peewee_repo import get_node
        node = get_node(m.get("hostname"))
        from lms.web.utils.asyncssh_utils import listdir
        data = await listdir(model_path + path, node=node)
        result = {"files": data}
        return json(result)

    @app.get("/lms/models/<model_name>/view")
    async def model_file_view(request, model_name):
        """
         查询 模型文件内容
        :param request:
        :param model_name:
        :param path:
        :return:
        """
        args = request.args
        path = args.get('path')
        model_name = unquote(model_name)
        m = query_model_by_model_name(model_name)
        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        model_path = m.get("model_path", "/")

        if model_path is None:
            raise Exception("model model_path: [{}] is not exists ! ".format(model_name))

        from lms.web.utils.peewee_repo import get_node
        node = get_node(m.get("hostname"))

        from lms.web.utils.asyncssh_utils import read_file_chunked

        async def streaming_fn(resp):
            async for data in read_file_chunked(model_path + path, node=node):
                await resp.write(data)

        content_type, encoding = mimetypes.guess_type(path, strict=False)
        return response.ResponseStream(streaming_fn,
                                       content_type="application/octet-stream" if content_type is None else content_type + "; charset=utf-8",
                                       headers={})

    @app.put("/lms/models/<model_name>/readme")
    async def update_readme(request, model_name):
        """
         更新readme.md 文件内容
        :param request:
        :param model_name:
        :return:
        """
        body = request.body
        model_name = unquote(model_name)
        m = query_model_by_model_name(model_name)
        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        model_path = m.get("model_path")

        if model_path is None:
            raise Exception("model model_path: [{}] is not exists ! ".format(model_name))
        from lms.web.utils.peewee_repo import get_node
        node = get_node(m.get("hostname"))
        await write_file(f'{model_path}/README.md', node=node, data=body)
        return json({})

    @app.get("/lms/models/<model_name>/restapi")
    async def query_restapi(request, model_name):
        """
         查询 模型调试接口
        :param request:
        :param model_name:
        :return:
        """
        model_name = unquote(model_name)

        m = query_model_by_model_name(model_name)
        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        if m['status'] == 'undeployed':
            raise NotFound()
        else:

            async with httpx.AsyncClient() as client:
                resp = await client.get(url=m['api_url'].replace('/prediction', '/prompt_role'))
            if resp.status_code == 200:
                prompt_role = resp.json()
            else:
                raise Exception("The server is not ok")

            async with httpx.AsyncClient() as client:
                resp = await client.get(url=m['api_url'].replace('/prediction', '/default_generate'))

            if resp.status_code == 200:
                generate = resp.json()
            else:
                raise Exception("The server is not ok")

            # if m['generate'] is None:
            #     request_body = {
            #         "messages": [
            #             {
            #                 "role": "User",
            #                 "content": "hello"
            #             }
            #         ],
            #         "repetition_penalty": 1.2,
            #         "top_k": 5,
            #         "top_p": 0.5,
            #         "temperature": 0.7,
            #         "max_new_tokens": 100
            #     }
            # else:
            #     import json as jzon
            #     request_body = jzon.loads(m['generate'])
            #     request_body.update({
            #         "messages": [
            #             {
            #                 "role": "User",
            #                 "content": "hello"
            #             }
            #         ]
            #     })
            #
            # result = {
            #     "api_url": m['api_url'],
            #     "api_key": m['api_key'],
            #     "headers": {
            #         "Content-Type": "application/json",
            #         "api_key": m['api_key']
            #     },
            #     "request_body": request_body
            # }

            request_body = {
                "messages": [{
                    "role": prompt_role["User"],
                    "content": "hello"
                }]
            }

            request_body.update(generate)

            result = {
                "api_url": m['api_url'],
                "api_key": m['api_key'],
                "headers": {
                    "Content-Type": "application/json",
                    "api_key": m['api_key']
                },
                "request_body": request_body,
                "prompt_role": prompt_role
            }
            return json(result)

    @app.get("/lms/models/<model_name>/evaluation/manual_eval")
    async def query_evaluation_manual_eval(request, model_name):
        """
         查询 手动评估数据
        :param request:
        :param model_name:
        :return:
        """
        model_name = unquote(model_name)

        m = query_model_by_model_name(model_name)
        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        return json(list_manual_eval(m['model_id']))

    @app.get("/lms/models/<model_name>/evaluation/scores")
    async def query_evaluation(request, model_name):
        """
        分页查询 评估数据
        :param request:
        :param model_name:
        :param state : all unevaluated evaluated :
        :return:
        """
        model_name = unquote(model_name)
        m = query_model_by_model_name(model_name)
        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        args = request.args
        state = args.get('state')
        current_page = int(args.get('current_page', 1))
        per_pages = int(args.get('per_pages', 10))
        manual_eval_id = args.get('manual_eval_id')

        return json(paging_eval_score(m['model_id'], state, current_page, per_pages, manual_eval_id))

    @app.put("/lms/models/<model_name>/evaluation/scores")
    async def update_evaluation(request, model_name):
        """
         提交手动评估数据
        :param request:
        :param model_name:
        :return:
        """
        request_json = request.json
        model_name = unquote(model_name)
        m = query_model_by_model_name(model_name)
        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        update_score(m['model_id'], request_json['manual_eval_id'], request_json['sn'], request_json['score'])
        return json({})

    @app.get("/lms/models/<model_name>/evaluation/metrics")
    async def query_metrics(request, model_name):
        """
         查询评估结果返回文本内容
        :param request:
        :param model_name:
        :return:
        """
        model_name = unquote(model_name)
        m = query_model_by_model_name(model_name)

        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        return json(query_metrics_by_model(m))

    @app.get("/lms/models/<model_name>/evaluation/leaderboard")
    async def query_leaderboard(request, model_name):
        model_name = unquote(model_name)
        check_file = os.path.isfile('./open_sota_model_metrics.json')

        if check_file:
            config_file = './open_sota_model_metrics.json'
        else:
            config_file = os.path.dirname(os.path.realpath(__file__)) + '/default_open_sota_model_metrics.json'

        with open(config_file) as f:
            data = f.read()
        import json as json0
        raw_data = json0.loads(data, object_pairs_hook=OrderedDict)
        m = query_model_by_model_name(model_name)
        if m is None:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        dd = query_metrics_by_model(m)
        raw_data.append(dd)

        benchmarks = OrderedDict()
        model_names = []
        models = {}
        for ent in raw_data:
            model_names.append(ent['model_name'])
            models[ent['model_name']] = {bm['benchmark_name']: bm['metrics'] for bm in ent['benchmarks']}
            for benchmark in ent['benchmarks']:
                benchmark_metrics = benchmarks.get(benchmark['benchmark_name'], list())
                unique_combine(benchmark_metrics, benchmark['metrics'].keys())
                benchmarks[benchmark['benchmark_name']] = benchmark_metrics

        model_names = model_names[-1:] + model_names[:-1]

        headers = ['benchmark', 'metrics']
        headers.extend(model_names)
        rows = []
        for benchmark, metrics in benchmarks.items():
            for metric in metrics:
                row = [benchmark, metric]
                for model_name in model_names:
                    value = models.get(model_name).get(benchmark, {}).get(metric, '-')
                    row.append(value)
                rows.append(row)

        result = {
            "headers": headers,
            "rows": rows
        }
        return json(result)

    @app.get("/lms/monitoring/metrics")
    async def query_monitoring_metrics(request):
        """
         查询 最新的模型cpu, gpu, memory和请求数
        :param request:
        :return:
        """
        from lms.web.utils import peewee_repo
        total = {
            "gpu_usage_percent": 0,
            "cpu_percent": 0,
            "mem_usage": 0
        }
        metrics = []

        model_dict = {m['model_name']: m for m in peewee_repo.query_model_list(status='deployed')}
        for node in peewee_repo.list_node():
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url="http://%s:%s/metrics" % (node['hostname'], node['daemon_port']))

                if resp.status_code == 200:
                    for metric in resp.json()['metrics']:
                        if metric['model_name'] in model_dict:
                            # filter deploying models
                            metrics.append(metric)
                            total['gpu_usage_percent'] = total['gpu_usage_percent'] + metric.get('gpu_usage_percent', 0)
                            total['cpu_percent'] = total['cpu_percent'] + metric.get('cpu_percent', 0)
                            total['mem_usage'] = total['mem_usage'] + metric.get('mem_usage', 0)
                else:
                    print(resp.status_code)
            except ConnectionError:
                pass
        result = {"metrics": metrics, "total": total}
        return json(result)

    @app.get("/lms/resources/total")
    async def query_resources(request):
        """
         查询集群资源总数
        :param request:
        :return:
        """

        total_resources = {"gpu_count": 10, "cpu_count": 20, "total_memory": 4096}

        result = {"total_resources": total_resources}

        return json(result)

    @app.post("/admin/pubkey")
    async def pubkey(request):
        id_rsa_pub = os.path.expanduser("~") + '/.ssh/id_rsa.pub'
        if os.path.isfile(id_rsa_pub):
            with open(id_rsa_pub, "r") as f:
                authorized_keys = f.read()
                result = {"pubkey": authorized_keys, "server_id": server_id}
                return json(result)
        else:
            raise Exception('You should generate SSH public key on the server')

    @app.post("/admin/join")
    async def join(request):
        request_json = request.json
        from lms.web.utils import peewee_repo
        if peewee_repo.get_node(request_json['hostname']) is None:
            peewee_repo.insert_node(request_json)
        else:
            peewee_repo.update_node(request_json)
            # raise Exception('The server was already joined by the node:%s' % request_json['hostname'])
        result = {}
        return json(result)

    @app.post("/admin/reset")
    async def reset(request):
        request_json = request.json
        from lms.web.utils import peewee_repo
        peewee_repo.remove_node(request_json)
        result = {}
        return json(result)

    @app.get("/admin/nodes")
    async def nodes(request):
        from lms.web.utils import peewee_repo
        result = {
            "nodes": peewee_repo.list_node()
        }
        return json(result)

    @app.post("/proxy/<server>/prediction")
    async def proxy(request, server):
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method=request.method,
                url="http://%s/prediction" % server,
                data=request.body,
                headers=request.headers,
                timeout=1000
            )
        return response.HTTPResponse(resp.content, headers=resp.headers, status=resp.status_code)

    @app.exception(NotFound)
    async def not_fount(request, e):
        import sys
        stacktrace = traceback_utils.get_tarceback(*sys.exc_info())
        traceback.print_exception(*sys.exc_info())
        return json({
            'code': -1,
            'message': str(e),
            'stacktrace': stacktrace
        }, status=404)

    @app.after_server_start
    async def listener(app, loop):
        print("started")

    @app.middleware('request')
    async def handle_request(request):
        """
         请求时打开数据库连接
        :param request:
        :return:
        """
        db.connect(reuse_if_open=True)

    @app.middleware('response')
    async def handle_response(request, response):
        """
        返回响应时关闭数据库连接
        :param request:
        :param response:
        :return:
        """
        if not db.is_closed():
            db.close()

    @app.exception(Exception)
    async def handle_exception(request, e):
        import sys
        stacktrace = traceback_utils.get_tarceback(*sys.exc_info())
        traceback.print_exception(*sys.exc_info())
        return json({
            'code': -1,
            'message': str(e),
            'stacktrace': stacktrace
        }, status=409)

    return app
