import datetime
import operator
import os
import uuid

import peewee
from peewee import Model as PeeweeModel, CharField, IntegerField, SqliteDatabase, ForeignKeyField, TextField, \
    DoubleField, \
    CompositeKey, DateTimeField, SQL, fn, Case

db_file_path = os.environ['database']

db = SqliteDatabase(db_file_path)


def create_database_tables():
    """
     创建表 如果存在则不会创建
    :return:
    """
    db.create_tables(
        [Model, MonitoringMetric, EvalMetric, ManualEval, EvalScore, Node],
        safe=True)


class Node(PeeweeModel):
    """
      手动评估数据表(lms_eval_score)
    """
    node_id = CharField(verbose_name='手动评估id', max_length=255, primary_key=True)
    hostname = CharField(verbose_name='主机名', max_length=255, unique=True)
    username = CharField(verbose_name='用户名', max_length=100, null=True)
    port = IntegerField(verbose_name='端口', null=False)
    daemon_port = IntegerField(verbose_name='监控端口', null=False)

    class Meta:
        database = db
        table_name = "lms_node"


class Model(PeeweeModel):
    """
       模型表(lms_model)
    """
    model_id = CharField(verbose_name='模型id', max_length=255, primary_key=True)
    model_name = CharField(verbose_name='模型名称', max_length=500, unique=True)
    model_path = CharField(verbose_name='模型路径', max_length=1000, unique=True)
    node = ForeignKeyField(Node, backref='models')
    size = CharField(verbose_name='模型大小', max_length=255, null=True)
    precision = CharField(verbose_name='模型量化标准', max_length=100, null=True)
    status = CharField(verbose_name='模型部署状态', max_length=100, null=True)
    api_url = CharField(verbose_name='模型测试url', max_length=1000, null=True)
    api_key = CharField(verbose_name='模型测试key', max_length=255, null=True)
    generate = CharField(verbose_name='模型默认参数', max_length=1024, null=True)
    deployment_time = DateTimeField(null=True)
    create_time = DateTimeField(null=True, default=datetime.datetime.now)
    last_update_time = DateTimeField(null=True, default=datetime.datetime.now)

    class Meta:
        database = db
        table_name = "lms_model"


class MonitoringMetric(PeeweeModel):
    """
      模型监控表(lms_monitoring_metric)
    """
    model_id = CharField(verbose_name='模型大ID', max_length=255, null=True)
    metric_name = CharField(verbose_name='指标名称', max_length=255, null=True)
    timestamp = DateTimeField(null=True, default=datetime.datetime.now)
    value = DoubleField(null=True)

    class Meta:
        database = db
        table_name = "lms_monitoring_metric"
        primary_key = False


class EvalMetric(PeeweeModel):
    """
      模型评估表(lms_eval_metric)
    """
    model_id = CharField(verbose_name='模型大ID', max_length=255, null=True)
    benchmark = TextField(verbose_name='评估参数', null=True)
    metric_name = CharField(verbose_name='指标名称', max_length=255, null=True)
    eval_kind = CharField(verbose_name='评估方式 automatic、custom、manual', max_length=255, null=True)
    value = DoubleField(null=True)

    class Meta:
        database = db
        table_name = "lms_eval_metric"
        primary_key = False


class ManualEval(PeeweeModel):
    """
      模型与评估数据关联表(lms_model_manual_eval)
    """
    manual_eval_id = CharField(verbose_name='手动评估id', max_length=255, primary_key=True)
    model_id = CharField(verbose_name='模型大ID', max_length=255, null=True)
    input_data_path = TextField(verbose_name='输入路径', null=True)
    output_data_path = TextField(verbose_name='输出路径', null=True)
    create_time = DateTimeField(null=True, default=datetime.datetime.now)

    class Meta:
        database = db
        table_name = "lms_model_manual_eval"
        constraints = [SQL('UNIQUE (model_id, input_data_path)')]


class EvalScore(PeeweeModel):
    """
      手动评估数据表(lms_eval_score)
    """
    manual_eval = ForeignKeyField(ManualEval, backref='eval_scores')
    sn = IntegerField(verbose_name='序号', null=False)
    input = TextField(verbose_name='输入', null=True)
    output = TextField(verbose_name='输出', null=True)
    expected = TextField(verbose_name='期望值', null=True)
    score = DoubleField(null=True)

    class Meta:
        database = db
        table_name = "lms_eval_score"
        primary_key = CompositeKey('manual_eval', 'sn')


@db.atomic()
def insert_model(**options):
    """
     插入模型数据
    :param options:
    :return:
    """
    m = Model()
    m.model_id = str(uuid.uuid1())

    node = Node.get(Node.hostname == options.get("hostname"))
    m.node = node

    if "model_id" in options:
        m.model_id = options.get("model_id")

    if "model_name" in options:
        m.model_name = options.get("model_name")

    if "model_path" in options:
        m.model_path = options.get("model_path")

    if "size" in options:
        m.size = options.get("size")

    if "precision" in options:
        m.precision = options.get("precision")

    if "status" in options:
        m.status = options.get("status")

    m.save(force_insert=True)


def query_model_by_model_name(model_name):
    """
    根据名称名称查询模型
    :param model_name:
    :return:
    """
    m = Model.get_or_none(Model.model_name == model_name)

    if m:
        return {'model_id': m.model_id,
                'model_name': m.model_name,
                'model_path': m.model_path,
                'hostname': m.node.hostname,
                'size': m.size,
                'precision': m.precision,
                'generate': m.generate,
                'status': m.status,
                'api_url': m.api_url,
                'api_key': m.api_key,
                'deployment_time': None if not m.deployment_time else m.deployment_time.strftime(
                    "%Y-%m-%d %H:%M:%S.%f"),
                'create_time': None if not m.create_time else m.create_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                'last_update_time': None if not m.last_update_time else m.last_update_time.strftime(
                    "%Y-%m-%d %H:%M:%S.%f")}

    return None


@db.atomic()
def delete_model_by_model_name(model_name):
    """
     根据模型名称删除模型
    :param model_name:
    :return:
    """
    m = Model.get_or_none(Model.model_name == model_name)
    if m:
        m.delete_instance()
        return True
    return False


@db.atomic()
def delete_model_by_model_id(model_id):
    """
    根据模型id删除模型
    :param model_id:
    :return:
    """
    m = Model.get_or_none(Model.model_id == model_id)
    if m:
        m.delete_instance()
        return True
    return False


@db.atomic()
def update_model_status_by_name(status, model_name):
    """
    根据模型名称 修改模型部署状态
    :param status:
    :param model_name:
    :return:
    """
    m = Model.get_or_none(Model.model_name == model_name)
    if m:
        m.status = status
        m.deployment_time = datetime.datetime.now()
        m.save()
        return True
    return False


def query_model_list(model_path=None, hostname=None, status=None):
    """
    查询模型列表 返回所有模型
    :return:
    """
    models = []

    model_select = Model.select()

    clauses = []
    if hostname is not None:
        model_select = model_select.join(Node)
        clauses.append((Node.hostname == hostname))
    if model_path is not None:
        clauses.append(Model.model_path == model_path)
    if status is not None:
        clauses.append(Model.status == status)

    if len(clauses) > 0:
        model_select = model_select.where(peewee.reduce(operator.and_, clauses))

    model_select = model_select.order_by(Model.create_time.desc())
    for m in model_select:
        models.append({'model_id': m.model_id,
                       'model_name': m.model_name,
                       'model_path': m.model_path,
                       'hostname': m.node.hostname,
                       'size': m.size,
                       'precision': m.precision,
                       'status': m.status,
                       'api_url': m.api_url,
                       'api_key': m.api_key,
                       'deployment_time': None if not m.deployment_time else m.deployment_time.strftime(
                           "%Y-%m-%d %H:%M:%S.%f"),
                       'create_time': None if not m.create_time else m.create_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                       'last_update_time': None if not m.last_update_time else m.last_update_time.strftime(
                           "%Y-%m-%d %H:%M:%S.%f")})
    return models


@db.atomic()
def update_model(model_name, **options):
    """
    根据模型名称 修改模型的值
    :param model_name:
    :param options:
    :return:
    """

    m = Model.get_or_none(Model.model_name == model_name)

    if m:
        m.last_update_time = datetime.datetime.now()
        if "model_path" in options:
            m.model_path = options.get("model_path")
        if "size" in options:
            m.size = options.get("size")
        if "precision" in options:
            m.precision = options.get("precision")
        if "status" in options:
            m.status = options.get("status")
        if "api_url" in options:
            m.api_url = options.get("api_url")
        if "api_key" in options:
            m.api_key = options.get("api_key")
        if "deployment_time" in options:
            m.deployment_time = options.get("deployment_time")
        if "generate" in options:
            import json as jzon
            m.generate = jzon.dumps(options.get("generate"))
        m.save()
        return True
    return False


@db.atomic()
def insert_eval_metrics(request_dict, m):
    print(request_dict)
    for benchmark in request_dict["benchmarks"]:
        for k, v in benchmark["metrics"].items():
            em = EvalMetric()
            em.model_id = m["model_id"]
            em.benchmark = benchmark['benchmark_name']
            em.eval_kind = request_dict['eval_kind']
            em.metric_name = k
            em.value = v
            em.save(force_insert=True)


def query_metrics_by_model(m):
    model_id = m['model_id']
    model_name = m['model_name']
    benchmarks = {}
    for m in EvalMetric.select().where(EvalMetric.model_id == model_id):
        metrics = benchmarks.get(m.benchmark, {})
        metrics[m.metric_name] = m.value
        benchmarks[m.benchmark] = metrics
    return {
        "model_name": model_name,
        "benchmarks": [{"benchmark_name": k, "metrics": v} for k, v in benchmarks.items()]
    }


@db.atomic()
def insert_manual_eval(request_dict, m, df):
    manual_eval = ManualEval()
    manual_eval.manual_eval_id = str(uuid.uuid1())
    manual_eval.model_id = m['model_id']
    manual_eval.input_data_path = request_dict['input_data_path']
    manual_eval.output_data_path = request_dict['output_data_path']
    manual_eval.save(force_insert=True)
    # 解析output_data_path入库
    scores = []
    for index, row in df.iterrows():
        score = EvalScore(sn=index,
                          manual_eval=manual_eval,
                          output=row['Output'],
                          input=row['input'],
                          expected=row['Expected'])
        scores.append(score)
    EvalScore.bulk_create(scores)


@db.atomic()
def update_manual_eval(request_dict, m, df):
    clauses = [(ManualEval.model_id == m['model_id']),
               (ManualEval.input_data_path == request_dict['input_data_path'])]
    manual_eval = ManualEval.select().where(peewee.reduce(operator.and_, clauses)).get()

    EvalScore.delete().where(EvalScore.manual_eval == manual_eval).execute()

    manual_eval.output_data_path = request_dict['output_data_path']
    manual_eval.create_time = datetime.datetime.now()
    manual_eval.save()

    scores = []
    for index, row in df.iterrows():
        score = EvalScore(sn=index,
                          manual_eval=manual_eval,
                          output=row['Output'],
                          input=row['input'],
                          expected=row['Expected'])
        scores.append(score)
    EvalScore.bulk_create(scores)


def paging_eval_score(model_id, state, current_page, per_pages, manual_eval_id):
    progress = ManualEval.select(ManualEval.manual_eval_id,
                                 fn.COUNT().alias('all'),
                                 fn.SUM(Case(None, ((EvalScore.score.is_null(True), 0),
                                                    (EvalScore.score.is_null(False), 1)), 0)).alias(
                                     'evaluated'),
                                 ).join(EvalScore).where(
        (ManualEval.model_id == model_id) &
        (ManualEval.manual_eval_id == manual_eval_id)).get()

    summary_stats = {
        "eval_progress": {
            "all_records_num": progress.all,
            "evaluated_records_num": progress.evaluated,
            "unevaluated_records_num": (progress.all - progress.evaluated)
        }
    }

    data = []

    clauses = [(ManualEval.model_id == model_id),
               (ManualEval.manual_eval_id == manual_eval_id)]
    if state == 'evaluated':
        clauses.append((EvalScore.score.is_null(False)))
    elif state == 'unevaluated':
        clauses.append((EvalScore.score.is_null(True)))

    total_records = EvalScore.select(fn.COUNT().alias("total_records")).join(ManualEval) \
        .where(peewee.reduce(operator.and_, clauses)).get().total_records

    pagination = {
        "total_records": total_records,
        "current_page": current_page
    }

    for m in EvalScore.select().join(ManualEval) \
            .where(peewee.reduce(operator.and_, clauses)) \
            .order_by(EvalScore.sn.asc()) \
            .paginate(current_page, per_pages):
        data.append({'sn': m.sn, 'input': m.input, 'expected': m.expected, 'output': m.output, 'score': m.score})

    return {
        'summary_stats': summary_stats,
        'pagination': pagination,
        'data': data
    }


def list_manual_eval(model_id):
    query = ManualEval.select(ManualEval.manual_eval_id,
                              fn.AVG(EvalScore.score).alias('score'),
                              fn.COUNT().alias('all'),
                              fn.SUM(Case(None, ((EvalScore.score.is_null(True), 0),
                                                 (EvalScore.score.is_null(False), 1)), 0)).alias(
                                  'evaluated'),
                              ).join(EvalScore).where(
        ManualEval.model_id == model_id).group_by(ManualEval.manual_eval_id)

    results = query.execute()
    map1 = {}
    for row in results:
        map1[row.manual_eval_id] = {
            'eval_progress': {
                'all_records_num': row.all,
                'evaluated_records_num': row.evaluated,
                'unevaluated_records_num': (row.all - row.evaluated)
            },
            'score': round(row.score, 2) if row.score is not None else row.score
        }

    data = []
    for m in ManualEval.select().where(ManualEval.model_id == model_id):
        dd = {'manual_eval_id': m.manual_eval_id,
              'data_path': m.input_data_path,
              # 'input_data_path': m.input_data_path,
              # 'output_data_path': m.output_data_path,
              'eval_time': m.create_time.strftime("%Y-%m-%d %H:%M:%S.%f")
              }
        dd.update(map1.get(m.manual_eval_id))
        data.append(dd)
    return {'evals': data}


def update_score(model_id, manual_eval_id, sn, score):
    for m in EvalScore.select().join(ManualEval).where(
            (ManualEval.manual_eval_id == manual_eval_id) &
            (EvalScore.sn == sn) &
            (ManualEval.model_id == model_id)
    ):
        m.score = score
        m.save()


def insert_node(request_dict):
    node = Node()
    node.node_id = str(uuid.uuid1())
    node.username = request_dict['username']
    node.hostname = request_dict['hostname']
    node.port = request_dict['port']
    node.daemon_port = request_dict['daemon_port']
    node.save(force_insert=True)


def update_node(request_dict):
    node = Node.get_or_none(Node.hostname == request_dict['hostname'])
    node.username = request_dict['username']
    node.port = request_dict['port']
    node.daemon_port = request_dict['daemon_port']
    node.save()


def remove_node(request_dict):
    Node.delete().where(Node.hostname == request_dict['hostname']).execute()


def get_node(hostname):
    return Node.get_or_none(Node.hostname == hostname)


def list_node():
    nodes = []
    for m in Node.select():
        nodes.append({
            "username": m.username,
            "hostname": m.hostname,
            "port": m.port,
            "daemon_port": m.daemon_port
        })
    return nodes
