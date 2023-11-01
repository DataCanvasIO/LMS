import argparse
import math
import signal
import sys
from secrets import token_hex

from lms.client import help_constants
from lms.client.common_utils import fill_missing
from lms.client.validaters import ValidateModelPath, ValidatePathExists


def check_model_name_or_path(parser, args):
    if not args.__contains__("model_name") and not args.__contains__("model_path"):
        parser.error("either --model_name or --model_path is required.")


def main():
    parser = argparse.ArgumentParser(prog='lms',
                                     description="""
                                         LMS is an open source tool that provides large model services. \
                                         LMS can provide model compression, model evaluation, model deployment, \
                                         model monitoring and other functions.
                                         """)

    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1.0"
    )

    subparsers = parser.add_subparsers(dest='sub_command', help='sub-command help')

    # join
    parser_join = subparsers.add_parser('join', help='To attach the client to Web server')
    parser_join.add_argument('server', type=str, help='Entrypoint of Web server. e.g. HOSTNAME:PORT')
    parser_join.add_argument('--token', type=str, required=False, help='Token for Web server to authenticate')
    parser_join.add_argument('--entrypoint', type=str, required=False,
                             help='Entrypoint of client which will be access by web server. e.g. HOSTNAME')

    # reset
    parser_reset = subparsers.add_parser('reset', help='To detach the client from Web server')

    # import
    # eg: lms import --model_path xx --model_name xx
    parser_import = subparsers.add_parser('import', help='To add a model to the Web server')
    parser_import.add_argument('--model_path', type=str, required=True, action=ValidateModelPath,
                               help=help_constants.MODEL_PATH)
    parser_import.add_argument('--model_name', type=str, help='Model name which will be registered in the server')

    # del
    # eg: lms del --model_path xx
    parser_del = subparsers.add_parser('del', help='To delete a model from the Web server')
    parser_del.add_argument('--model_path', type=str, action=ValidateModelPath,
                            help=help_constants.MODEL_PATH)
    parser_del.add_argument('--model_name', type=str, help=help_constants.MODEL_NAME)

    # list
    # eg: lms list --model_path xx
    parser_list = subparsers.add_parser('list', help='To list models')

    # eval
    # eg: lms eval --model_path xx --task xx --output_path xx --input_path xx
    parser_eval = subparsers.add_parser('eval', help='To evaluate model')
    parser_eval.add_argument('--model_path', type=str, action=ValidateModelPath,
                             help=help_constants.MODEL_PATH)
    parser_eval.add_argument('--model_name', type=str, help=help_constants.MODEL_NAME)
    parser_eval.add_argument('--input_path', type=str, action=ValidatePathExists, help='Input data')
    parser_eval.add_argument('--output_path', type=str, help='Output data')
    parser_eval.add_argument('--task', type=str, required=True, help="""Task name of evaluation.
    Automatic Evaluation: 
        Specifying names in MMLU, CMMLU, BigBench, ARC, AGIEval, ceval separated by comma
    Custom evaluation: 
        Specifying a Python file with suffix '.py'
    Manual Evaluation: 
        Other of above
    """)

    # quantization
    # lms quantization --model_path xx --int8 --output_model_name xx --output_path xx
    parser_quant = subparsers.add_parser('quantization', help='To quantize model')
    parser_quant.add_argument('--model_path', type=str, action=ValidateModelPath,
                              help=help_constants.MODEL_PATH)
    parser_quant.add_argument('--model_name', type=str, help=help_constants.MODEL_NAME)
    parser_quant.add_argument('--quantized_model_path', type=str,
                              help='Specify the directory to save the result model.')
    parser_quant.add_argument('--int8', action='store_true',
                              help='Specify the quantization precision to int8, and the Bitsandbytes will be use')
    parser_quant.add_argument('--int4', action='store_true',
                              help='Specify the quantization precision to int4, and the GPTQ will be use')

    # pruning
    # lms pruning --model_path xx --method xx --pruned_model_path  xx
    parser_pruning = subparsers.add_parser('pruning', help='To prune model')

    pruning_subparsers = parser_pruning.add_subparsers(dest='sub_pruning_command', help="sub-sub help")
    # for sparse
    parser_pruning_sparse = pruning_subparsers.add_parser('sparse')
    parser_pruning_sparse.add_argument('--model_path', type=str, action=ValidateModelPath,
                                       help=help_constants.MODEL_PATH)
    parser_pruning_sparse.add_argument('--model_name', type=str, help=help_constants.MODEL_NAME)
    parser_pruning_sparse.add_argument('--pruned_model_path', type=str, required=True,
                                       help='Specify the directory to save the result model.')
    parser_pruning_sparse.add_argument('--seed', type=int, help='The random seed to select calibration samples.')
    parser_pruning_sparse.add_argument('--device', type=str,
                                       help='Device to use. For example: cpu, cuda, cuda:0, cuda:1, etc.')
    parser_pruning_sparse.add_argument('--layer_name_start', type=str, required=False,
                                       help='Specify the layer name from which to start pruning.')
    parser_pruning_sparse.add_argument('--layer_name_stop', type=str, required=False,
                                       help='Specify the layer name util which to stop pruning. Will include this layer.')
    parser_pruning_sparse.add_argument('--nsamples', type=int, default=64,
                                       help='The number of samples used for pruning.')
    parser_pruning_sparse.add_argument('--seqlen', type=int, default=1024,
                                       help='The sequence length of each sample tokens.')
    parser_pruning_sparse.add_argument('--sparsity', type=str, default="2:4",
                                       help='sparsity: The sparse ratio or the n:m sparsity paradigm.\nFor example: 0.6 to specify the sparse ratio of each layer,\n2:4 to specify 2 out of every 4 elements are zero.')

    # for structure
    parser_pruning_structure = pruning_subparsers.add_parser('structure')
    parser_pruning_structure.add_argument('--model_path', type=str, action=ValidateModelPath,
                                          help='Model dir. Only support llama, vicuna, bloom, baichuan models.')
    parser_pruning_structure.add_argument('--model_name', type=str, help=help_constants.MODEL_NAME)
    parser_pruning_structure.add_argument('--pruned_model_path', type=str, required=True,
                                          help=' Specify the directory to save the result model.')
    parser_pruning_structure.add_argument('--seed', type=int, help='Seed.')
    parser_pruning_structure.add_argument('--device', type=str,
                                          help='Device to use. For example: cpu, cuda, cuda:0, cuda:1, etc.')
    parser_pruning_structure.add_argument('--model_type', type=str, choices=['llama', 'vicuna', 'bloom', 'baichuan'],
                                          required=True,
                                          help='One of llama,vicuna,bloom,baichuan')
    parser_pruning_structure.add_argument('--pruning_ratio', type=float, default=0.5, help='Pruning ratio')
    parser_pruning_structure.add_argument('--pruner_type', type=str, default='l2', choices=['l2', 'l1', 'taylor'],
                                          help='One of l2, l1, taylor.')
    parser_pruning_structure.add_argument('--block_attention_layer_start', type=int, required=False,
                                          help='Start layer of block attention layers')
    parser_pruning_structure.add_argument('--block_attention_layer_end', type=int, required=False,
                                          help='End layer of block attention layers')
    parser_pruning_structure.add_argument('--block_mlp_layer_start', type=int, required=False,
                                          help='Start layer of block mlp layers')
    parser_pruning_structure.add_argument('--block_mlp_layer_end', type=int, required=False,
                                          help='End layer of block mlp layers')

    # deploy
    # lms deploy --model_path xx --model_type{{gpt2/bloom/llama/gpt-neox}} --port xx --gpu 0,1 \
    # --load_{{fp16/int8/int4}} --num_processor xx --infer_config infer_conf.json
    parser_deploy = subparsers.add_parser('deploy', help='To deploy model')
    parser_deploy.add_argument('--model_path', type=str, action=ValidateModelPath, help=help_constants.MODEL_PATH)
    parser_deploy.add_argument('--model_name', type=str, help=help_constants.MODEL_NAME)
    parser_deploy.add_argument('--port', type=int, required=False, help='Specifying port. If not present, '
                                                                        'you will get a random port')
    parser_deploy.add_argument('--timeout', type=int, default=1800, required=False, help='Deployment timeout')
    parser_deploy.add_argument('--load_fp16', action='store_true', help='Loading as fp16')
    parser_deploy.add_argument('--load_int8', action='store_true', help='Loading as int8')
    parser_deploy.add_argument('--load_int4', action='store_true', help='Loading as int4')
    parser_deploy.add_argument('--api_key', type=str, help='Specifying API key. If not present, '
                                                           'you will got a random key')
    parser_deploy.add_argument('--loglevel', type=str, default='INFO', choices=['ERROR', 'WARN', 'INFO', 'DEBUG'],
                               help='Log level')
    parser_deploy.add_argument('--infer_py', type=str, action=ValidatePathExists, help='A Python file to load '
                                                                                       'a pipeline manually')
    parser_deploy.add_argument('--infer_config', type=str, help='Setting default model parameters')
    parser_deploy.add_argument('--gpu', type=str,
                               help='List of GPU devices to bind to with comma separated list. e.g. 0,2')
    # parser_deploy.add_argument('--model_type', type=str, help='global_params help')
    # parser_deploy.add_argument('--num_processor', type=str, help='global_params help')
    # parser_deploy.add_argument('--method', type=str, help='global_params help')

    # logs
    parser_logs = subparsers.add_parser('logs', help='To show logs of deployed model')
    parser_logs.add_argument('--model_path', type=str, action=ValidateModelPath, help=help_constants.MODEL_PATH)
    parser_logs.add_argument('--model_name', type=str, help=help_constants.MODEL_NAME)
    parser_logs.add_argument('-f', '--follow', default=False, action='store_true',
                             help='Output appended data as the file grows;')

    # undeploy
    # lms undeploy --model_path xx
    parser_undeploy = subparsers.add_parser('undeploy', help='To undeploy model')
    parser_undeploy.add_argument('--model_path', action=ValidateModelPath, type=str, help=help_constants.MODEL_PATH)
    parser_undeploy.add_argument('--model_name', type=str, help=help_constants.MODEL_NAME)

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    argv = sys.argv

    if args.sub_command == "import":
        from lms.client.import_module import add
        add(model_path=args.model_path, model_name=args.model_name)

    elif args.sub_command == "del":
        check_model_name_or_path(parser, args)
        model_path, model_name = fill_missing(args.model_path, args.model_name)
        from lms.client.import_module import delete
        delete(model_path=model_path, model_name=model_name)

    elif args.sub_command == "list":
        from lms.client.import_module import _list
        models = _list()

        def isCN(ch):
            if '\u4e00' <= ch <= '\u9fff':
                return True
            return False

        def length(chars):
            cnt = 0
            for ch in chars:
                if isCN(ch):
                    cnt += 1.7143
                else:
                    cnt += 1
            return math.ceil(cnt)

        widths = [length(m['model_name']) for m in models]
        widths.append(10)
        width = max(widths)
        print(("{:<" + str(width) + "}\t{:<10}\t{:<19}\t{:<255}").format('MODEL_NAME', 'STATUS', 'CREATE_TIME',
                                                                         'MODEL_PATH'))
        for item in models:

            cn_number = 0
            for ch in item['model_name']:
                if isCN(ch):
                    cn_number += 0.7143
            align_width = width - cn_number
            print(("{:<" + str(align_width) + "}\t{:<10}\t{:<19}\t{:<255}").format(
                item['model_name'], item['status'], item['create_time'][:19],
                item['model_path']))

    elif args.sub_command == "deploy":
        check_model_name_or_path(parser, args)
        model_path, model_name = fill_missing(args.model_path, args.model_name)
        from lms.client.deploy_module import deploy
        if args.api_key is None:
            args.api_key = token_hex(16)

        def sigterm_handler(signal, frame):
            # save the state here or do whatever you want
            print('booyah! bye bye')
            sys.exit(0)

        signal.signal(signal.SIGTERM, sigterm_handler)

        def get_dtype(args):
            dtype = None
            if args.load_int4:
                dtype = 'int4'
            elif args.load_int8:
                dtype = 'int8'
            elif args.load_fp16:
                dtype = 'fp16'
            return dtype

        deploy(model_path=model_path, model_name=model_name, port=args.port, api_key=args.api_key,
               timeout=args.timeout, loglevel=args.loglevel, unknown=unknown, infer_config_path=args.infer_config,
               dtype=get_dtype(args), infer_py=args.infer_py, gpu=args.gpu)

    elif args.sub_command == "logs":
        check_model_name_or_path(parser, args)
        model_path, model_name = fill_missing(args.model_path, args.model_name)
        from lms.client.deploy_module import logs
        logs(model_path=model_path, model_name=model_name, follow=args.follow)

    elif args.sub_command == "undeploy":
        check_model_name_or_path(parser, args)
        model_path, model_name = fill_missing(args.model_path, args.model_name)
        from lms.client.deploy_module import undeploy
        undeploy(model_path=model_path, model_name=model_name)

    elif args.sub_command == "eval":
        check_model_name_or_path(parser, args)
        model_path, model_name = fill_missing(args.model_path, args.model_name)
        from lms.client.evaluate_module import evaluate
        evaluate(model_path, model_name, args, argv)
    elif args.sub_command == "quantization":
        check_model_name_or_path(parser, args)
        model_path, model_name = fill_missing(args.model_path, args.model_name)
        from lms.client.quantize_module import quantize
        quantize(model_path, model_name, args, argv)
    elif args.sub_command == "pruning":
        if args.sub_pruning_command is None:
            parser_pruning.print_help()
            parser.error("missing arguments")

        check_model_name_or_path(parser, args)
        model_path, model_name = fill_missing(args.model_path, args.model_name)
        from lms.client.prune_module import prune
        prune(model_path, model_name, args, argv)
    elif args.sub_command == "join":
        from lms.client.cluster.adm import join
        join(args.server, args.token, args.entrypoint)
    elif args.sub_command == "reset":
        from lms.client.cluster.adm import reset
        reset()
    else:
        parser.error("lms: missing arguments\nTry `lms --help' for more information.")


if __name__ == "__main__":
    main()
