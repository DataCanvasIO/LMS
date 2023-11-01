import datetime
import logging
import os
import uuid
from enum import Enum
from urllib.parse import unquote

import aiofiles as aiofiles
import numpy as np
import pandas as pd
from peewee import IntegrityError
from sanic import Blueprint
from sanic.response import json

from lms.web.utils.peewee_repo import query_model_by_model_name, insert_model, delete_model_by_model_id, \
    update_model, insert_manual_eval, insert_eval_metrics, query_model_list, update_manual_eval


class Model_Status(Enum):
    DEPLOYED = "deployed"
    UNDEPLOYED = "undeployed"


logger = logging.getLogger(__name__)


def make_models():
    models = Blueprint(name="internal")

    def verify_model_by_name(model_name):
        return query_model_by_model_name(model_name)

    @models.post("/lms/internal/models")
    async def add_model(request):
        """
         添加模型接口
        :param request:
        :return:
        """
        request_dict = request.json
        model_name = request_dict.get("model_name")

        if "model_name" not in request_dict or model_name is None:
            raise Exception("model model_name cannot be empty ! ")

        logger.info("request : {}".format(request_dict))

        m = verify_model_by_name(model_name)
        if m:
            raise Exception("model model_name: [{}] already exists ! ".format(model_name))

        model_id = str(uuid.uuid1())
        request_dict['model_id'] = model_id
        request_dict['status'] = Model_Status.UNDEPLOYED.value

        # 入库操作
        insert_model(**request_dict)

        return json({"model_name": model_name, "model_id": model_id})

    @models.delete("/lms/internal/models/<model_name>")
    async def remove_model(request, model_name):
        """
         删除模型接口
        :param request:
        :param model_name:
        :return:
        """
        model_name = unquote(model_name)
        # 根据模型名称查询模型是否存在
        m = query_model_by_model_name(model_name)
        if not m:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        # 删除模型
        delete_model_by_model_id(m['model_id'])

        return json({"model_name": model_name, "model_id": m['model_id']})

    @models.post("/lms/internal/models/<model_name>/deployment")
    async def deployment_model(request, model_name):
        """
         部署模型接口
        :param request:
        :param model_name:
        :return:
        """

        options = request.json
        model_name = unquote(model_name)
        # 根据模型名称查询模型是否存在
        m = query_model_by_model_name(model_name)
        if not m:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        # if Model_Status.DEPLOYED.value == m.get("status"):
        #     raise Exception("model model_name: [{}] is already deployed !".format(model_name))

        # 修改数据库模型状态
        if not options:
            options = {"deployment_time": datetime.datetime.now()}
        else:
            options['deployment_time'] = datetime.datetime.now()
        options['status'] = Model_Status.DEPLOYED.value
        update_model(model_name, **options)

        return json({})

    @models.delete("/lms/internal/models/<model_name>/deployment")
    async def cancel_deployment_model(request, model_name):
        """
         取消部署模型接口
        :param request:
        :param model_name:
        :return:
        """
        model_name = unquote(model_name)

        # 根据模型名称查询模型是否存在
        m = query_model_by_model_name(model_name)
        if not m:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        if Model_Status.UNDEPLOYED.value == m.get("status"):
            raise Exception("model model_name: [{}] is not deployed !".format(model_name))

        options = {"status": Model_Status.UNDEPLOYED.value}
        update_model(model_name, **options)

        return json({})

    @models.post("/lms/internal/models/<model_name>/evaluation")
    async def evaluation_model(request, model_name):
        """
         评估模型接口
        :param request:
        :param model_name:
        :return:
        """
        model_name = unquote(model_name)

        # 根据模型名称查询模型是否存在
        m = query_model_by_model_name(model_name)
        if not m:
            raise Exception("model model_name: [{}] is not exists ! ".format(model_name))

        request_dict = request.json

        if request_dict['eval_kind'] == 'human':
            from lms.web.utils.peewee_repo import get_node
            node = get_node(m.get("hostname"))
            output_data_path = request_dict['output_data_path']
            from lms.web.utils.asyncssh_utils import read_file_chunked
            local_path = "./" + str(uuid.uuid1())
            try:
                async with aiofiles.open(local_path, mode='wb') as f:
                    async for data in read_file_chunked(output_data_path[output_data_path.index(":") + 1:], node=node,
                                                        chunk_size=4096):
                        await f.write(data)

                df = pd.read_csv(local_path).replace({np.nan: None})
                try:
                    insert_manual_eval(request_dict, m, df)
                except IntegrityError:
                    update_manual_eval(request_dict, m, df)
            finally:
                os.remove(local_path)
        else:
            insert_eval_metrics(request_dict, m)
        return json({})

    @models.get("/lms/internal/models")
    async def get_models(request):
        args = request.args
        model_path = args.get('model_path')
        hostname = args.get('hostname')
        data = query_model_list(model_path=model_path, hostname=hostname)
        result = {'models': data}
        return json(result)

    return models
