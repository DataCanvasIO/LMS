import argparse
import json as jzon
import logging
import os
import re
import sys
import traceback
from json import JSONDecodeError

from lms.runtime.common import traceback_utils
from lms.runtime.common.common_utils import read_infer_config

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2'))
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Deepspeed Inference")
    parser.add_argument("--model_path", type=str, required=True, help="")
    parser.add_argument("--model_name", type=str, help="")
    parser.add_argument("--learning_type", type=str, default="text-generation", help="")
    parser.add_argument('--local_rank', type=int, required=True, help='global_params help')
    parser.add_argument("--dtype", type=str, help="", default=None)
    parser.add_argument("--device", type=str, help="", default=None)
    parser.add_argument("--logfile", type=str, help="", default=None)
    parser.add_argument("--infer_config", type=str, help="", default=None)
    parser.add_argument('--server_id', type=str, help='')
    args = parser.parse_args()

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile, level=logging.DEBUG)

    import deepspeed
    import torch
    import torch.distributed as dist

    from lms.runtime.common.common_utils import load_pipeline_for_deepspeed

    dist.init_process_group(backend='nccl')

    infer_config = read_infer_config(args.infer_config)

    tokenizer_kwargs = infer_config.get('tokenizer_kwargs', {})
    model_kwargs = infer_config.get('model_kwargs', {})
    # tokenizer_kwargs = {
    #     "padding_side": "left"
    # }
    pipe = load_pipeline_for_deepspeed(args.model_path, tokenizer_kwargs=tokenizer_kwargs, model_kwargs=model_kwargs,
                                       learning_type=args.learning_type)

    pipe.model = deepspeed.init_inference(
        pipe.model,
        mp_size=world_size,
        dtype=torch.float
    )

    pipe.device = torch.device(f'cuda:{local_rank}')

    if not dist.is_initialized() or dist.get_rank() == 0:
        sys.stderr.write("CMD:STARTED\n")
        while True:
            try:
                content = input("prompt:")
                match = re.match(r'^CMD:REQ:(?P<content>.*)$', content)
                if match:
                    _content = match.group("content")
                    try:
                        model_args = jzon.loads(_content)
                        print("*************")
                        print(model_args)
                        object_list = [{
                            "content": model_args[1],
                            "model_kwargs": model_args[2]
                        }]

                        dist.broadcast_object_list(object_list, src=0)
                        _content = object_list[0]['content']
                        _model_kwargs = object_list[0]['model_kwargs']
                        output = pipe(_content, **_model_kwargs)
                        print(output)
                        print(f"\nCMD:RESP:{jzon.dumps([model_args[0], 0, output])}")
                    except JSONDecodeError:
                        logger.warning("found an invalid line: %s", content)
                    except Exception as e:
                        stacktrace = traceback_utils.get_tarceback(*sys.exc_info())
                        output = {
                            'code': -1,
                            'message': str(e),
                            'stacktrace': stacktrace
                        }
                        print(f"\nCMD:RESP:{jzon.dumps([model_args[0], -1, output])}")
                else:
                    object_list = [{
                        "content": content,
                        "model_kwargs": {}
                    }]

                    dist.broadcast_object_list(object_list, src=0)
                    _content = object_list[0]['content']
                    _model_kwargs = object_list[0]['model_kwargs']
                    output = pipe(_content, **_model_kwargs)
                    print(jzon.dumps(output))
            except EOFError:
                pass
    else:
        while True:
            try:
                object_list = [None]
                dist.broadcast_object_list(object_list, src=0)
                _content = object_list[0]['content']
                _model_kwargs = object_list[0]['model_kwargs']
                pipe(_content, **_model_kwargs)
            except Exception:
                traceback.print_exc()


if __name__ == '__main__':
    main()
