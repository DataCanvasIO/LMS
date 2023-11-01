import asyncio
import contextlib
import logging
import re
import sys
import traceback
import uuid

import fastjsonschema
import torch

from lms.runtime.common.common_utils import read_infer_config, load_config
from lms.runtime.deploy.request import PredictionCountedRequest

MAXSIZE = 10
logger = logging.getLogger(__name__)


class CaseInsensitiveDict(dict):
    """Case-insensitive dictionary implementation."""

    def __init__(self, pairs=None):
        if pairs is not None:
            for k, v in pairs.items():
                self.__setitem__(k, v)

    def __getitem__(self, key):
        return dict.__getitem__(self, key.casefold())

    def __setitem__(self, key, value):
        return dict.__setitem__(self, key.casefold(), value)

    def __delitem__(self, key):
        return dict.__delitem__(self, key.casefold())


def get_traceback(type, value, tb):
    return '\n'.join(str(x) for x in traceback.format_exception(type, value, tb))


with open(__file__.replace("serving.py", "request_body.json"), 'rb') as f:
    import json as _json

    schema_validate = fastjsonschema.compile(_json.load(f))


def make_app(model_path, model_name, api_key, loglevel, logfile, device, infer_config_path, dtype, infer_py, server_id,
             args):
    if logfile is not None:
        logging.basicConfig(filename=logfile, level=logging.getLevelName(loglevel))
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(loglevel))
    from sanic import Sanic, exceptions
    from sanic.response import json
    app = Sanic(name="default", request_class=PredictionCountedRequest)

    infer_config = read_infer_config(infer_config_path)
    learning_type = "text-generation"

    _prompt_role = infer_config.get('prompt_role', {
        "User": "User",
        "Assistant": "Assistant"
    })

    _generate = infer_config.get('generate', {
        "repetition_penalty": 1.2,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 0.6,
        "max_new_tokens": 1024,
    })

    async def generate_task(app):
        try:
            async def read_stdout(stdout, ctx):
                while True:
                    line = await stdout.readline()
                    if not line:
                        break
                    import json as jzon
                    match = re.match(r'^CMD:RESP:(?P<content>.*?)\n$', str(line, 'utf-8'))
                    if match:
                        content = match.group("content")
                        logger.debug("found answer: %s", content)
                        try:
                            resp_array = jzon.loads(content)
                        except jzon.decoder.JSONDecodeError:
                            logger.warning("found an invalid line: %s", str(line, 'utf-8'))
                            continue
                        request_id = resp_array[0]
                        # result = resp_array[1]
                        resp_queue = ctx.resp_queue_dict.get(request_id, None)
                        if resp_queue:
                            await resp_queue.put(resp_array)
                    else:
                        sys.stderr.write(str(line, 'utf-8'))

            async def read_stderr(stderr, ctx):
                while True:
                    line = await stderr.readline()
                    if not line:
                        break
                    if line == bytes('CMD:STARTED\n', 'utf-8'):
                        ctx.ready = True

                    sys.stderr.write(str(line, 'utf-8'))

            async def write_stdin(stdin, ctx):
                while True:
                    (request_id, string, kwargs) = await ctx.req_queue.get()
                    logger.debug("found question: %s", [request_id, string, kwargs])
                    import json as jzon
                    buf = f'CMD:REQ:{jzon.dumps([request_id, string, kwargs])}\n'.encode()
                    stdin.write(buf)
                    await stdin.drain()

            async def check_status(proc):
                while True:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(proc.wait(), 1e-6)
                    if proc.returncode is not None:
                        raise Exception("The Inference process was exited unexpectedly")
                    else:
                        await asyncio.sleep(3)
                        continue

            #
            config = load_config(model_path)
            env = {}
            if config is None or \
                    args.gpu is None or \
                    (config is not None and config.get("quantization_config", None) is not None) or \
                    (dtype == 'int4' or dtype == 'int8') or \
                    infer_py is not None:
                # transformers inference
                logger.info("Using Transformers inference")
                import lms.runtime.deploy.transformers_infer as transformers_infer
                infer_args = [transformers_infer.__file__,
                              f'--model_path={model_path}']

                if model_name is not None:
                    infer_args.append(f'--model_name={model_name}')

                if server_id is not None:
                    infer_args.append(f'--server_id={server_id}')

                if learning_type is not None:
                    infer_args.append(f'--learning_type={learning_type}')

                if device is not None:
                    infer_args.append(f'--device={device}')

                if dtype is not None:
                    infer_args.append(f'--dtype={dtype}')

                if infer_config_path is not None:
                    infer_args.append(f'--infer_config={infer_config_path}')

                if infer_py is not None:
                    infer_args.append(f'--infer_py={infer_py}')

            else:
                # deepspeed inference
                logger.info("Using Deepspeed inference")

                def find_free_port():
                    import socket
                    from contextlib import closing
                    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                        s.bind(('', 0))
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        return s.getsockname()[1]

                device_count = torch.cuda.device_count()
                for i in args.gpu.split(','):
                    if int(i) < 0 or int(i) >= device_count:
                        raise Exception('CUDA error: invalid device ordinal:' + i)

                env['CUDA_VISIBLE_DEVICES'] = args.gpu

                def encode_world_info(world_info):
                    import json as _json
                    import base64 as _base64
                    world_info_json = _json.dumps(world_info).encode('utf-8')
                    world_info_base64 = _base64.urlsafe_b64encode(world_info_json).decode('utf-8')
                    return world_info_base64

                gpus = [int(i) for i in args.gpu.split(',')]
                import lms.runtime.deploy.deepspeed_infer as deepspeed_infer
                infer_args = [
                    '-u', '-m', 'deepspeed.launcher.launch',
                    f'--world_info={encode_world_info({"localhost": gpus})}',
                    '--master_addr=127.0.0.1',
                    f'--enable_each_rank_log=None',
                    f"--master_port={str(find_free_port())}",
                    deepspeed_infer.__file__,
                    f'--model_path={model_path}']

                if model_name is not None:
                    infer_args.append(f'--model_name={model_name}')

                if server_id is not None:
                    infer_args.append(f'--server_id={server_id}')

                if learning_type is not None:
                    infer_args.append(f'--learning_type={learning_type}')

                if device is not None:
                    infer_args.append(f'--device={device}')

                if dtype is not None:
                    infer_args.append(f'--dtype={dtype}')

                if infer_config_path is not None:
                    infer_args.append(f'--infer_config={infer_config_path}')

            if logfile is not None:
                infer_args.append(f'--logfile={logfile}')

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                *infer_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            logger.info("The inference process:%s has been started." % proc)
            await asyncio.gather(
                read_stderr(proc.stderr, app.ctx),
                read_stdout(proc.stdout, app.ctx),
                write_stdin(proc.stdin, app.ctx),
                check_status(proc)
            )
        except Exception:
            traceback.print_exception(*sys.exc_info())
            app.stop()
            sys.exit(1)

    @app.listener('after_server_start')
    def create_task_queue(app, loop):
        app.ctx.req_queue = asyncio.Queue(loop=loop, maxsize=MAXSIZE)
        app.ctx.resp_queue_dict = {}
        app.ctx.loop = loop
        app.ctx.ready = False
        app.add_task(generate_task(app))

    @app.get("/prompt_role")
    async def prompt_role(request: PredictionCountedRequest):
        return json(_prompt_role)

    @app.get("/default_generate")
    async def default_generate(request: PredictionCountedRequest):
        return json(_generate)

    @app.post("/prediction")
    async def prediction(request: PredictionCountedRequest):
        body = request.json
        schema_validate(body)

        segments = []
        for entry in request.json['messages']:
            if entry['role'] in _prompt_role.values():
                segments.append(f"{entry['role']}: {entry['content']} \n")
            else:
                raise Exception(f"Unsupported role: {entry['role']}, supported roles:{list(_prompt_role.values())}")
        segments.append(_prompt_role.get('Assistant') + ':')
        content = "".join(segments)

        del body['messages']
        if 'repetition_penalty' in body:
            body['repetition_penalty'] = float(body['repetition_penalty'])

        kwargs = infer_config.get('generate', {})
        kwargs.update(body)

        request_id = str(uuid.uuid4())
        resp_queue = asyncio.Queue(loop=request.app.ctx.loop, maxsize=MAXSIZE)
        request.app.ctx.resp_queue_dict[request_id] = resp_queue
        try:
            await request.app.ctx.req_queue.put((request_id, content, kwargs))
            result = await resp_queue.get()
        finally:
            request.app.ctx.resp_queue_dict.pop(request_id, None)

        code = result[1]
        output = result[2]
        if code == 0:
            if output and type(output) is list and output[0] \
                    and output[0].get('generated_text', None) \
                    and output[0].get('generated_text').startswith(content):
                return json({"content": [{'generated_text': output[0].get('generated_text')[len(content):]}]})
            else:
                return json({"content": output})
        else:
            return json(output, status=409)

    @app.get("/metrics")
    async def metrics(request: PredictionCountedRequest):
        return json({"total_requests": request.count})

    @app.get("/ready")
    async def ready(request):
        return json({'ready': request.app.ctx.ready})

    @app.middleware("request")
    async def validate(request):
        if not hasattr(request.app.ctx, 'ready') or not request.app.ctx.ready:
            raise exceptions.ServiceUnavailable()

        if request.path == '/prediction':
            _api_key = CaseInsensitiveDict(request.headers).get('api_key', None)
            if _api_key is None:
                raise exceptions.Unauthorized()
            if _api_key != api_key:
                raise exceptions.Forbidden()

    @app.exception(exceptions.SanicException)
    async def handle_sanic_exception(request, e):
        stacktrace = get_traceback(*sys.exc_info())
        return json({
            'code': -1,
            'message': str(e),
            'stacktrace': stacktrace
        }, status=e.status_code)

    @app.exception(Exception)
    async def handle_exception(request, e):
        stacktrace = get_traceback(*sys.exc_info())
        traceback.print_exception(*sys.exc_info())
        return json({
            'code': -1,
            'message': str(e),
            'stacktrace': stacktrace
        }, status=409)

    return app
