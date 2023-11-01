import re
import subprocess


def nvml_pmon():
    """
    nvmlDeviceGetProcessUtilization
    https://www.clear.rice.edu/comp422/resources/cuda/pdf/nvml.pdf
    :return:
    """
    cmd = "nvidia-smi pmon -c 1 -s mu"
    try:
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        # output = "# gpu        pid  type    fb    sm   mem   enc   dec   command\n# Idx          #   C/G    MB     %     %     %     %   name\n    0      25985     C   893     0     0     -     -   python3.7\n    0      92760     C   893     12     0     -     -   python3.7"
        # print(output)
        headers = None
        stats = {}
        for i, line in enumerate(output.splitlines()):
            if line.startswith("#"):
                if i == 0:
                    headers = re.split(r"\s+", line.strip())[1:]
                continue
            values = re.split(r"\s+", line.strip())
            dict0 = dict(zip(headers, values))
            stats[dict0["pid"]] = dict0
        return stats
    except subprocess.CalledProcessError as e:
        return {}


if __name__ == '__main__':
    print(nvml_pmon())
