# Quick Guide

## 1. Introduction

LMS is an open source tool that provides large model services. LMS can provide model compression, model evaluation,
model deployment, model monitoring and other functions. It includes LMS_Web and LMS_Client. In LMS_Client, you can
import, compress and deploy models through the command line. LMS_Web provides basic model information and a visual
interface for monitoring deployed models.

## 2. Guide

### 2.1 Installation

You can install LMS with the following command.

```bash
pip install lms
```

or

```bash
git clone git@gitlab.datacanvas.com:APS-OpenSource/lms.git
cd lms
python setup.py install
```

### 2.2 Start & Stop lms_web

You can start LMS_Web with the following command.

```bash
lms_web start --port {{LMS_WEB_PORT}}
```

You can stop LMS_Web with the following command.

```bash
lms_web stop
```

### 2.3 Setting

You should initiate client using following command which will register self into the web center and start a daemon to
expose the monitoring metrics.

```bash
lms join {{LMS_WEB_HOST_NAME}}:{{LMS_WEB_PORT}}
```

### 2.4 Model List

1. Add model
   You can add the model to LMS through the following command.

    ```bash
    lms import --model_path {{model_path}}
    ```

   After adding the model successfully,You can view the model in LMS_Web.

2. Delete model
   You can delete the model from LMS through the following command.

    ```bash
    lms del --model_name {{model_name}}
    ```

3. List model
   You can list the model of the current node from the LMS with the following command.

    ```bash
    lms list
    ```

### 2.5 Model Evaluation

1. Automatic evaluation:
   the preset task will be used to test the specified model. The preset task includes `MMLU`, `CMMLU`, `BigBench`, `ARC`
   , `AGIEval`, `ceval` and other benchmark. The model will be evaluated from multiple angles. You can automatically
   evaluate the model with the following instructions.
    ```bash
    lms eval --model_name {{model_name}} --task CMMLU,ceval,ARC --output_path {{output_path}}
    ```

2. Custom evaluation
   The system will use your specified task to test the specified model, and you can make a custom evaluation with the
   following command.

    ```bash
    lms eval --model_name {{model_name}} --task {{custom_task.py}} --input_path {{input_path}} --output_path {{output_path}}
    ```

3. Manual evaluation
   Execute the following command, and the system will test the model with the data you specify and return the model
   results to the specified path.

    ```bash
    lms eval --model_name {{model_name}} --task human --input_path {{input_path}} --output_path {{output_path}}
    ```

   After the execution is successful, the model output will be displayed on the LMS_Web corresponding model details
   page. You need to manually evaluate the output of the model.

After the evaluation is successful, the evaluation result will be displayed on the LMS_Web corresponding model details
page.

### 2.6 Quantization

You can quantize the specified model with the following command.

```bash
lms quantization --model_name {{model_name}} --{{int8|int4}} --quantized_model_path {{quantized_model_path}}
```

> See examples for more details.

After the quantization of the model is completed, a new quantized model is generated, and you can view the new model in
LMS_Web.

### 2.7 Pruning

You can prune the specified model with the following command.

```bash
lms pruning {{sparse|structure}} --model_name {{model_name}}  --pruned_model_path {{pruned_model_path}}
```

After the pruning of the model is completed, a new pruned model is generated, and you can view the new model in LMS_Web.

### 2.8 Model Deployment

You can deploy bloom, llama, falcon, and many other models using the following commands.

1. deploy supported model
    ```bash
    lms deploy --model_name {{model_name}} --gpu 0,1 --load_{{fp16|int8|int4}} --infer_config infer_conf.json
    ```

   If you need to deploy unsupported model, you can use deploy custom model.

2. deploy custom model
    ```bash
    lms deploy --model_name {{model_name}} --gpu 0,1 --load_{{fp16|int8|int4}} --infer_py generate.py --infer_config infer_conf.json
    ```

> See examples for more details on infer_config and infer_py.

If you want to deploy with specific port, use flag:`--port {{port}}` at above commands.

After the model is deployed successfully, you can view the deployment status and monitoring of the model in LMS_Web.

1. undeploy model

   You can undeploy the mode with the following command.

    ```bash
    lms undeploy --model_name {{model_name}}
    ```

2. deployment logs

   If you want to watch the log of the deployed model, you can use the following command.
   ```bash
   lms logs -f --model_name {{model_name}}
   ```

## 3. Compatibility matrix

| ModelName | Quantize int8 | Quantize int4 | Prune sparse | Prune structure | Deploy fp16 | Deploy int8 | Deploy int4 |
|-----------|---------------|---------------|--------------|-----------------|-------------|-------------|-------------|
| Alaya    | Y             | Y             | Y            | Y               | Y           | Y           | Y           |
| llama2    | Y             | Y             | Y            | Y               | Y           | Y           | Y           |
| RedPajama | Y             | Y             | Y            | ❌               | Y           | Y           | Y           |
| chatyuan  | Y             | ❌             | ❌            | ❌               | Y           | Y           | Y           |
| bloom     | Y             | Y             | Y            | Y               | Y           | Y           | Y           |
| falcon    | Y             | Y             | Y            | ❌               | Y           | Y           | Y           |
| mpt       | Y             | Y             | Y            | ❌               | Y           | Y           | Y           |
| GPT-J     | Y             | ❌             | Y            | ❌               | Y           | Y           | Y           |
| dolly     | Y             | Y             | Y            | ❌               | Y           | Y           | Y           |
| T5        | Y             | ❌             | ❌            | ❌               | Y           | Y           | Y           |


