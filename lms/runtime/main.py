import argparse
import json
import re
import sys

import torch

from lms.runtime.evaluation.eval_module import evaluate1
from lms.runtime.quantize.bnb import bnb_quantize
from lms.runtime.quantize.gptq import gptq_quantize


# lms deploy --model_path xx --model_type{{gpt2/bloom/llama/gpt-neox}} --port xx --gpu 0,1 --load_{{fp16/int8/int4}} --num_processer xx --infer_config infer_conf.json
def main():
    parser = argparse.ArgumentParser(prog='lms' if sys.argv[0].endswith('lms_rt') else sys.argv[0],
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

    # deploy
    # lms deploy --model_path xx --model_type{{gpt2/bloom/llama/gpt-neox}} --port xx --gpu 0,1 \
    # --load_{{fp16/int8/int4}} --num_processor xx --infer_config infer_conf.json
    parser_deploy = subparsers.add_parser('deploy', help='deploy help222')
    parser_deploy.add_argument('--model_path', type=str, required=True, help='global_params help')
    parser_deploy.add_argument('--model_name', type=str, help='global_params help')
    parser_deploy.add_argument('--model_type', type=str, help='global_params help')
    parser_deploy.add_argument('--port', type=int, required=True, help='global_params help')
    parser_deploy.add_argument('--gpu', type=str, help='global_params help')
    parser_deploy.add_argument('--dtype', type=str, default='fp16', choices=['int4', 'int8', 'fp16']
                               , help='global_params help')
    parser_deploy.add_argument('--num_processor', type=str, help='global_params help')
    parser_deploy.add_argument('--method', type=str, help='global_params help')
    parser_deploy.add_argument('--infer_config', type=str, help='global_params help')
    parser_deploy.add_argument('--loglevel', type=str, default='INFO', choices=['ERROR', 'WARN', 'INFO', 'DEBUG'],
                               help='global_params help')
    parser_deploy.add_argument('--logfile', type=str, help='global_params help')
    parser_deploy.add_argument('--api_key', type=str, default='default', help='global_params help')
    parser_deploy.add_argument('--infer_py', type=str, help='global_params help')
    parser_deploy.add_argument('--server_id', type=str, help='global_params help')

    # eval
    # eg: lms eval --model_path xx --task xx --output_path xx --input_path xx --eval_py main.py
    parser_eval = subparsers.add_parser('eval', help='eval help222')
    parser_eval.add_argument('--model_path', type=str, required=True, help='global_params help')
    parser_eval.add_argument('--task', type=str, required=True, help='global_params help')
    parser_eval.add_argument('--input_path', type=str, help='global_params help')
    parser_eval.add_argument('--output_path', type=str, required=True, help='global_params help')

    # quantization
    # lms quantization --model_path xx --int8 --pruned_model_path xx
    parser_quant = subparsers.add_parser('quantization', help='quantization help')
    parser_quant.add_argument('--model_path', type=str, help='Model dir')
    parser_quant.add_argument('--model_name', type=str, help='Model name which registered in the server')
    parser_quant.add_argument('--quantized_model_path', type=str,
                              help='Specify the directory to save the result model.')
    parser_quant.add_argument('--int8', action='store_true',
                              help='Specify the quantization precision to int8, and the Bitsandbytes will be use')
    parser_quant.add_argument('--int4', action='store_true',
                              help='Specify the quantization precision to int4, and the GPTQ will be use')

    # pruning
    # lms pruning --model_path xx --method xx --pruned_model_path  xx
    parser_pruning = subparsers.add_parser('pruning', help='pruning help')

    pruning_subparsers = parser_pruning.add_subparsers(dest='sub_pruning_command', help="sub-sub help")
    # for sparse
    parser_pruning_sparse = pruning_subparsers.add_parser('sparse')
    parser_pruning_sparse.add_argument('--model_path', type=str, help='Model dir')
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
                                       help='sparsity: The sparse ratio or the n:m sparsity paradigm.\n'
                                            'For example: 0.6 to specify the sparse ratio of each layer,\n'
                                            '2:4 to specify 2 out of every 4 elements are zero.')

    # for structure
    parser_pruning_structure = pruning_subparsers.add_parser('structure')
    parser_pruning_structure.add_argument('--model_path', type=str,
                                          help='Model dir. Only support llama, vicuna, bloom, baichuan models.')
    parser_pruning_structure.add_argument('--pruned_model_path', type=str, required=True,
                                          help='Specify the directory to save the result model.')
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

    args, unknown = parser.parse_known_args()

    if args.sub_command == "deploy":
        from lms.runtime.deploy.serving import make_app
        make_app(args.model_path, args.model_name, args.api_key, args.loglevel, args.logfile,
                 'cpu' if args.gpu is None else "cuda:" + args.gpu,
                 args.infer_config, args.dtype, args.infer_py, args.server_id, args). \
            run(host='0.0.0.0', port=args.port, single_process=True)
        # from sanic import Sanic
        # from sanic.worker.loader import AppLoader
        # from functools import partial
        # loader = AppLoader(
        #     factory=partial(make_app, args.model_path, args.model_name, args.api_key, args.loglevel, args.logfile,
        #                     'cpu' if args.gpu is None else "cuda:" + args.gpu,
        #                     args.infer_config, args.dtype, args.infer_py, args))
        # app = loader.load()
        # app.prepare(host="0.0.0.0", port=args.port, single_process=True)
        # Sanic.serve(primary=app, app_loader=loader)


    elif args.sub_command == "eval":
        if args.task == 'human':
            if args.input_path is None:
                parser.error("--task=human requires --input_path")
        elif re.match(r"^.*\.py$", args.task):
            if args.input_path is None:
                parser.error("--task=custom requires --input_path")
        else:
            accepts = ['ARC', 'MMLU', 'CMMLU', 'ceval', 'AGIEval', 'BigBench']
            tasks = args.task.split(",")
            for task in tasks:
                if task not in accepts:
                    parser.error("The task:'%s' is not in %s on automatic evaluation" % (task, accepts))

        output = evaluate1(args.model_path,
                           args.task,
                           args.input_path,
                           args.output_path)
        print(json.dumps(output))
    elif args.sub_command == "quantization":
        parser_quant.add_argument('--device', type=str, help='global_params help')
        args = parser.parse_args()
        if args.device is not None:
            device = args.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if args.int8:
            bnb_quantize(model_path=args.model_path, model_save_path=args.quantized_model_path, device=device)
        elif args.int4:
            gptq_quantize(model_path=args.model_path, model_save_path=args.quantized_model_path, device=device)
        else:
            parser.error("either --int8 or --fp16 is required")
    elif args.sub_command == "pruning":
        if args.sub_pruning_command == 'sparse':
            if args.device is not None:
                device = args.device
            else:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

            tokenizer_path = None
            only_check_model = False
            from lms.runtime.prune.sparsegpt import prune
            prune(model_path=args.model_path,
                  tokenizer_path=tokenizer_path,
                  only_check_model=only_check_model,
                  layer_name_start=args.layer_name_start,
                  layer_name_stop=args.layer_name_stop,
                  seed=args.seed,
                  nsamples=args.nsamples,
                  seqlen=args.seqlen,
                  sparsity=args.sparsity,
                  model_save_path=args.pruned_model_path,
                  tokenizer_save_path=args.pruned_model_path,
                  device=device
                  )
        elif args.sub_pruning_command == 'structure':
            from lms.runtime.prune.llm_pruner import prune
            if args.device is not None:
                device = args.device
            else:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

            prune(args.model_path,
                  args.model_type,
                  args.pruned_model_path,
                  args.pruning_ratio,
                  args.pruner_type,
                  args.block_attention_layer_start,
                  args.block_attention_layer_end,
                  args.block_mlp_layer_start,
                  args.block_mlp_layer_end,
                  device,
                  eval_before_prune=False,
                  eval_after_prune=False,
                  seed=args.seed)
        else:
            raise Exception("")


if __name__ == "__main__":
    main()
