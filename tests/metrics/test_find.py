# from deepspeed.launcher.runner import main
import re

# LMS_RT_PATTERN = r'.*(lms_rt\s+deploy|deepspeed_infer\.py|transformers_infer\.py).*--model_path=.*?\s+--model_name=(?P<model_name>?)\s+(--port=(?P<port>\d*))?(?!.*>)'
LMS_RT_PATTERN = r'.*(lms_rt\s+deploy|deepspeed_infer\.py|transformers_infer\.py).*--model_path=.*?\s+--model_name=(?P<model_name>.*?)\s+(--port=(?P<port>\d*))?\s*--server_id='

with open("/Users/dev/PycharmProjects/lms/tests/metrics/a.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        # print(line)
        m = re.match(LMS_RT_PATTERN, line)
        if m:
            print(m.group('model_name'))
            print(m.group('port'))
            print(line)
