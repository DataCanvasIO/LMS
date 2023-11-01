import random
from typing import Union

import torch


def load_tokenizer(tokenizer_path):
    def try_load(clz):
        try:
            tokenizer = clz.from_pretrained(tokenizer_path)
        except Exception as e:
            print(e)
            print("Failed to load tokenizer, try trust_remote_code=True.")
            tokenizer = clz.from_pretrained(tokenizer_path, trust_remote_code=True)
        return tokenizer

    if "llama" in tokenizer_path.lower():
        from transformers import LlamaTokenizer
        return try_load(LlamaTokenizer)
    else:
        from transformers import AutoTokenizer
        return try_load(AutoTokenizer)


def get_c4_batch(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )

    random.seed(seed)

    # For train
    inps = []
    tars = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:  # 只要大于seqlen的样本
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        inps.append(inp)
        tars.append(tar)

    train_batch = torch.concat(inps, dim=0), torch.concat(tars, dim=0)

    return train_batch[0]


def model_info(model, words=""):
    import torch
    assert isinstance(model, torch.nn.Module)
    if words == "":
        print(f"{'Param Info':=^200}")
    else:
        print(f"{words:=^200}")

    param_num = 0
    param_mem = 0
    for param_name, param in model.named_parameters():
        module = type(model.get_submodule(".".join(param_name.split(".")[:-1])))
        type_name = module.__module__ + "." + module.__name__
        if len(type_name) > 60:
            type_name = f"{type_name[:30]}...{type_name[-30:]}"

        param_num += param.numel()
        param_mem += param.nelement() * param.element_size()
        sparse_num = (param == 0).sum().item()
        sparse_ratio = sparse_num / param.numel()
        shape_str = f"{str(param.dtype):-<12s}---sparsity={sparse_ratio:.3f}---[{', '.join([str(_) for _ in param.shape])}]={param.numel()}"
        print(f"Param : {type_name:-<66s}---{param_name:-<66s}---{shape_str}")
    # https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2
    buf_num = 0
    buf_mem = 0
    for buf_name, buf in model.named_buffers():
        module = type(model.get_submodule(".".join(buf_name.split(".")[:-1])))
        type_name = module.__module__ + "." + module.__name__
        if len(type_name) > 60:
            type_name = f"{type_name[:30]}...{type_name[-30:]}"

        buf_num += buf.numel()
        buf_mem += buf.nelement() * buf.element_size()
        sparse_num = (buf == 0).sum().item()
        sparse_ratio = sparse_num / buf.numel()
        shape_str = f"{str(buf.dtype):-<12s}---sparsity={sparse_ratio:.3f}---[{', '.join([str(_) for _ in buf.shape])}]={buf.numel()}"
        print(f"Buffer: {type_name:-<66s}---{buf_name:-<66s}---{shape_str}")

    print(f"Param  numel: {param_num} is {param_num / 10 ** 9} Billion. GPU mem: {param_mem / 1024 ** 3} GB")
    if buf_num > 0:
        print(f"Buffer numel: {buf_num} is {buf_num / 10 ** 9} Billion. GPU mem: {buf_mem / 1024 ** 3} GB")
        total_num = param_num + buf_num
        total_mem = param_mem + buf_mem
        print(f"Total  numel: {total_num} is {total_num / 10 ** 9} Billion. GPU mem: {total_mem / 1024 ** 3} GB")

    print("End.", words, "\n\n")


class Node:
    visited = set()
    reused = set()

    def __init__(self, name, module, verbose=False):
        assert isinstance(module, torch.nn.Module)
        Node.visited.add(module)
        self.name = name
        self.module = module
        self.verbose = verbose

        self.parent = None
        self.children = []
        for k, v in module.named_children():
            if v in Node.visited:
                Node.reused.add(v)
                continue
            cur_node = Node(k, v, verbose=verbose)
            cur_node.parent = self
            self.children.append(cur_node)

    @property
    def full_name(self):
        if self.parent is None or self.parent.full_name == "":
            return self.name
        return self.parent.full_name + "." + self.name

    def __str__(self):
        clz = type(self.module)
        s = f"{self.full_name}: {clz.__module__}.{clz.__name__}"
        if self.verbose:
            try:
                next(iter(self.module.children()))
            except StopIteration:
                s += f": {self.module}"
        return s

    def __repr__(self):
        repr_str = str(self)

        for i, c in enumerate(self.children):
            for j, t in enumerate(repr(c).split("\n")):
                if i < len(self.children) - 1:
                    if "├── " in t or "│── " in t or "└── " in t:
                        repr_str += f"\n│   {t}"
                    else:
                        repr_str += f"\n├── {t}"
                else:
                    if j == 0:
                        repr_str += f"\n└── {t}"
                    else:
                        repr_str += f"\n    {t}"
        return repr_str


def recurse_to_device(elements: Union[list, tuple, dict, torch.Tensor], device, moved_info=None):
    if moved_info is None:
        moved_info = {"numel": 0, "memory": 0}
    if isinstance(elements, torch.Tensor):
        if elements.device != device:
            elements.data = elements.data.to(device)
            numel = elements.numel()
            moved_info["numel"] += numel
            moved_info["memory"] += elements.element_size() * numel

    elif isinstance(elements, (list, tuple)):
        for v in elements:
            recurse_to_device(v, device, moved_info=moved_info)
    elif isinstance(elements, dict):
        for v in elements.values():
            recurse_to_device(v, device, moved_info=moved_info)
    else:
        pass
    return moved_info


def recurse_tensor_info(elements: Union[list, tuple, dict, torch.Tensor], tensor_info=None):
    if tensor_info is None:
        tensor_info = {"numel": 0, "memory": 0}
    if isinstance(elements, torch.Tensor):
        numel = elements.numel() / 1024 ** 3
        tensor_info["numel"] += numel
        memory = elements.element_size() * numel
        tensor_info["memory"] += memory
        device = str(elements.device)

        if device not in tensor_info:
            tensor_info[device] = {"numel": 0, "memory": 0}
        tensor_info[device]["numel"] += numel
        tensor_info[device]["memory"] += memory

    elif isinstance(elements, (list, tuple)):
        for v in elements:
            recurse_tensor_info(v, tensor_info=tensor_info)
    elif isinstance(elements, dict):
        for v in elements.values():
            recurse_tensor_info(v, tensor_info=tensor_info)
    else:
        pass
    return tensor_info


class DeviceHook:
    def __init__(self, device, verbose=False):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        self.verbose = verbose
        from collections import defaultdict
        self.pre_device = defaultdict(list)
        self.forward_pre_handles = {}
        self.forward_handles = {}

        self._module2name = defaultdict(list)
        self._numel_in = 0
        self._numel_out = 0
        self._modules_in = []
        self._modules_out = []

    @property
    def _modules_on_device(self):
        on = self._modules_in.copy()
        on.reverse()
        for v in self._modules_out:
            on.remove(v)
        on.reverse()
        return on

    def forward_pre_hook(self, module, args, kwargs):
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
        assert isinstance(module, torch.nn.Module)
        recurse_to_device([args, kwargs], self.device)

        changed = False
        for n, p in module.named_parameters(recurse=False):
            if p.device == self.device:
                self.pre_device[(module, n)].append(None)
            else:
                self.pre_device[(module, n)].append(p.device)
                p.data = p.data.to(self.device)
                changed = True
                self._numel_in += p.numel()
        for n, b in module.named_buffers(recurse=False):
            if b.device == self.device:
                self.pre_device[(module, n)].append(None)
            else:
                self.pre_device[(module, n)].append(b.device)
                b.data = b.data.to(self.device)
                changed = True
                self._numel_in += b.numel()
        if changed:
            self._modules_in.append(module)

    def forward_hook(self, module, args, kwargs, output):
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
        assert isinstance(module, torch.nn.Module)
        if self.verbose:
            print(
                f"Device {self.device}: Allocated {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.3f}GB. At: {self._module2name[module]}")
            tensor_info = recurse_tensor_info([args, kwargs, output])
            print(f"       Activations info: {tensor_info}")

        changed = False
        for n, p in module.named_parameters(recurse=False):
            pre_device = self.pre_device[(module, n)].pop()
            if pre_device:
                p.data = p.data.to(pre_device)
                changed = True
                self._numel_out += p.numel()
        for n, b in module.named_buffers(recurse=False):
            pre_device = self.pre_device[(module, n)].pop()
            if pre_device:
                changed = True
                b.data = b.data.to(pre_device)
                self._numel_out += b.numel()
        if changed:
            self._modules_out.append(module)
            torch.cuda.empty_cache()
            if self.verbose:
                modules_on_device = self._modules_on_device
                if len(modules_on_device) > 0:
                    print(
                        f"       Empty cache: Allocated   {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.3f}GB. Modules on device: {modules_on_device}, elements:  {self._numel_in - self._numel_out}, ")
                else:
                    print(
                        f"       Empty cache: Allocated   {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.3f}GB. No modules on device.")

    def attach_to_module(self, module):
        assert isinstance(module, torch.nn.Module)
        seen = set()
        for k, v in module.named_modules():
            self._module2name[v].append(k)
            if v in seen:
                continue
            seen.add(v)
            self.forward_pre_handles[v] = v.register_forward_pre_hook(hook=self.forward_pre_hook, with_kwargs=True)
            self.forward_handles[v] = v.register_forward_hook(hook=self.forward_hook, with_kwargs=True)

    def remove_hooks(self):
        for v in (*self.forward_pre_handles.values(), *self.forward_handles.values()):
            v.remove()


# find layers
class ModuleFinder:
    class FinderStopped(Exception):
        pass

    def __init__(self, start, stop, classes=(torch.nn.Linear, torch.nn.Conv2d)):
        self.start = start
        self.stop = stop
        self.classes = classes
        self.start_flag = False
        self.matched_modules = []
        self.handles = []
        self.module2name = {}

    def trace(self, module, args):
        if self.start_flag:
            if isinstance(module, self.classes):
                self.matched_modules.append(module)
                print(f"Will prune this module: {self.module2name[module]}")

    def find_start(self, module, args):
        print("Start to find modules to prune...")
        self.start_flag = True

    def find_stop(self, module, args, output):
        self.start_flag = False
        print(f"{len(self.matched_modules)} modules will be pruned.")
        raise self.FinderStopped

    def attach_to_module(self, module):
        for k, v in module.named_modules():
            self.module2name[v] = k
            self.handles.append(v.register_forward_pre_hook(hook=self.trace))
            if k == self.start:
                self.handles.append(v.register_forward_pre_hook(hook=self.find_start, prepend=True))
            elif k == self.stop:
                self.handles.append(v.register_forward_hook(hook=self.find_stop))

    def get_matched_modules(self):
        for h in self.handles:
            h.remove()
        return self.matched_modules


class AddSparseHook:

    def __init__(self, module, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01, device=None):
        self.sparsity = sparsity
        self.prunen = prunen
        self.prunem = prunem
        self.blocksize = blocksize
        self.percdamp = percdamp

        assert isinstance(module, torch.nn.Module)
        self.handle = module.register_forward_hook(hook=self.add_batch, with_kwargs=True, prepend=True)

    def add_batch(self, module, args, kwargs, output):
        self.handle.remove()
        from .sparsegpt import SparseGPT
        sparse_wraped = SparseGPT(module)

        sparse_wraped.add_batch(args[0].data, output.data)

        print(f"Pruning... {module}")
        sparse_wraped.fasterprune(self.sparsity, prunen=self.prunen, prunem=self.prunem, blocksize=self.blocksize,
                                  percdamp=self.percdamp)
        sparse_wraped.free()
        del sparse_wraped
        print("           Finished!")
        del output
        new_output = module(*args, **kwargs)
        return new_output

    @staticmethod
    def attach_to_modules(modules, sparsity, prunen, prunem):
        for target_module in modules:
            AddSparseHook(target_module, sparsity=sparsity, prunen=prunen, prunem=prunem)
