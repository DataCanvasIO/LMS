import argparse
import json as jzon
import logging
import re
import sys
from json import JSONDecodeError

from lms.runtime.common import traceback_utils
from lms.runtime.common.common_utils import load_class_object, read_infer_config

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Transformer Inference")

    parser.add_argument("--model_path", type=str, required=True, help="")
    parser.add_argument("--model_name", type=str, help="")
    parser.add_argument("--learning_type", type=str, default="text-generation", help="")
    parser.add_argument("--dtype", type=str, help="", default=None)
    parser.add_argument("--device", type=str, help="", default=None)
    parser.add_argument("--logfile", type=str, help="", default=None)
    parser.add_argument("--infer_config", type=str, help="", default=None)
    parser.add_argument("--infer_py", type=str, help="", default=None)
    parser.add_argument('--server_id', type=str, help='')

    args = parser.parse_args()

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile, level=logging.DEBUG)

    infer_config = read_infer_config(args.infer_config)

    if args.infer_py is None:
        from lms.runtime.common.common_utils import load_pipeline
        tokenizer_kwargs = infer_config.get('tokenizer_kwargs', {})
        model_kwargs = infer_config.get('model_kwargs', {})
        # tokenizer_kwargs = {
        #     "padding_side": "left"
        # }
        # model_kwargs = {}
        pipe = load_pipeline(
            args.model_path, tokenizer_kwargs, model_kwargs, args.learning_type, args.dtype, args.device
        )
    else:
        custom_class_object = load_class_object(args.infer_py)
        pipe = custom_class_object.load(
            args.model_path,
            learning_type=args.learning_type.lower(),
            dtype=args.dtype,
            device=args.device,
        )

    sys.stderr.write("CMD:STARTED\n")
    while True:
        try:
            content = input("prompt:")
            match = re.match(r"^CMD:REQ:(?P<content>.*)$", content)
            if match:
                _content = match.group("content")
                try:
                    model_args = jzon.loads(_content)
                    print("*************")
                    print(model_args)
                    output = pipe(
                        model_args[1], **model_args[2]
                    )
                    print(output)
                    print(f"\nCMD:RESP:{jzon.dumps([model_args[0], 0, output])}")
                except JSONDecodeError:
                    logger.warning("found an invalid line: %s", content)
            else:
                result = pipe(content)
                print(jzon.dumps(result))
        except EOFError:
            pass
        except Exception as e:
            stacktrace = traceback_utils.get_tarceback(*sys.exc_info())
            output = {"code": -1, "message": str(e), "stacktrace": stacktrace}
            print(f"\nCMD:RESP:{jzon.dumps([model_args[0], -1, output])}")


if __name__ == "__main__":
    main()
