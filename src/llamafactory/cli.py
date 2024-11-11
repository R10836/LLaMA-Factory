# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import subprocess
import sys
from enum import Enum, unique

from . import launcher
from .api.app import run_api
from .chat.chat_model import run_chat
from .eval.evaluator import run_eval
from .extras import logging
from .extras.env import VERSION, print_env
from .extras.misc import get_device_count
from .train.tuner import export_model, run_exp
from .webui.interface import run_web_demo, run_web_ui


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli eval -h: evaluate models                        |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to LLaMA Factory, version {VERSION}"
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
    + "-" * 58
)

logger = logging.get_logger(__name__)


@unique
class Command(str, Enum):
    API = "api"
    CHAT = "chat"
    ENV = "env"
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    VER = "version"
    HELP = "help"


"""
1 - sys.argv是 Python 标准库 sys 模块中的一个列表，它包含了命令行参数。即使 main 函数没有显式地传递参数，sys.argv 仍然可以访问到命令行参数，因为它是全局的。
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml时sys.argv 列表会自动填充内容：['llamafactory-cli', 'train', 'examples/train_lora/llama3_lora_sft.yaml']
sys.argv[0] 是脚本的名称，即 'llamafactory-cli'。
sys.argv[1] 是第一个参数，即 'train'。
sys.argv[2] 是第二个参数，即 'examples/train_lora/llama3_lora_sft.yaml'。
sys.argv.pop(1) 会移除并返回第二个元素 'train'，将其赋值给 command 变量。此时 sys.argv 列表变为：['llamafactory-cli', 'examples/train_lora/llama3_lora_sft.yaml']。

2 - 检查环境变量 FORCE_TORCHRUN 是否设置为 true 或 1，或者设备数量是否大于 1。如果满足条件，则进行分布式训练初始化：
在分布式训练的情况下，torchrun 命令会将 sys.argv[1:] 作为参数传递给启动脚本。此时 sys.argv[1:] 为 ['examples/train_lora/llama3_lora_sft.yaml']。
在单机训练的情况下，run_exp() 函数会直接使用 sys.argv 中的参数。

3 - 'examples/train_lora/llama3_lora_sft.yaml' 这个参数在命令行中传递后，通过一系列函数调用和参数解析，最终被用于训练过程。以下是详细的参数流动过程：

命令行调用：

当你运行 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml 时，sys.argv 列表包含命令行参数：['llamafactory-cli', 'train', 'examples/train_lora/llama3_lora_sft.yaml']。
解析命令：

在 main 函数中，sys.argv.pop(1) 移除并返回第二个元素 'train'，将其赋值给 command 变量。此时 sys.argv 列表变为：['llamafactory-cli', 'examples/train_lora/llama3_lora_sft.yaml']。
处理 train 命令：

main 函数中匹配到 elif command == Command.TRAIN: 分支，检查是否需要进行分布式训练。
如果需要分布式训练，则使用 torchrun 命令进行分布式训练，并将 sys.argv[1:] 作为参数传递给启动脚本。
如果不需要分布式训练，则调用 run_exp() 函数。
调用 run_exp 函数：

run_exp 函数会读取 sys.argv[1]，即配置文件路径 'examples/train_lora/llama3_lora_sft.yaml'。
run_exp 函数调用 get_train_args(args)，其中 args 默认是 None。
解析训练参数：

get_train_args 函数调用 _parse_train_args(args)，传递 args 参数（默认为 None）。
_parse_train_args 函数创建一个 HfArgumentParser 对象，并调用 _parse_args(parser, args)。
解析 YAML 文件：

_parse_args 函数检查 sys.argv 列表，如果只有一个参数且以 .yaml 或 .yml 结尾，则调用 parser.parse_yaml_file 方法解析 YAML 文件。
解析后的参数被赋值给 model_args, data_args, training_args, finetuning_args, generating_args。
利用解析后的参数：

run_exp 函数根据 finetuning_args.stage 的值，调用相应的训练函数（如 run_sft）。
这些训练函数会使用解析后的参数来初始化模型、数据集和训练设置，并开始训练。

"""
def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.API:
        run_api()
    elif command == Command.CHAT:
        run_chat()
    elif command == Command.ENV:
        print_env()
    elif command == Command.EVAL:
        run_eval()
    elif command == Command.EXPORT:
        export_model()
    elif command == Command.TRAIN:
        force_torchrun = os.getenv("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
        if force_torchrun or get_device_count() > 1:
            master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
            master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))
            logger.info_rank0(f"Initializing distributed tasks at: {master_addr}:{master_port}")
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                ).format(
                    nnodes=os.getenv("NNODES", "1"),
                    node_rank=os.getenv("NODE_RANK", "0"),
                    nproc_per_node=os.getenv("NPROC_PER_NODE", str(get_device_count())),
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                ),
                shell=True,
            )
            sys.exit(process.returncode)
        else:
            run_exp() # run_exp() 函数会读取 sys.argv[1]，即配置文件路径 llama3_lora_sft.yaml。
    elif command == Command.WEBDEMO:
        run_web_demo()
    elif command == Command.WEBUI:
        run_web_ui() # 前端
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError(f"Unknown command: {command}.")
