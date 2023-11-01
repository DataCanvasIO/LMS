import subprocess
import threading

process = subprocess.Popen("python3 /Users/dev/PycharmProjects/lms/lms_rt/lms/rt/test/test1.py",
                           shell=True,
                           stdout=subprocess.PIPE,
                           # stderr=subprocess.STDOUT
                           )


def print_progress(stdout):
    from tqdm import tqdm
    tqdm.write(stdout.readline().decode("utf-8"))


t1 = threading.Thread(target=print_progress, args=(process.stdout,))
t1.start()
t1.join()

for line in iter(process.stdout.readline, b''):
    # print(str(line, 'UTF-8'))
    line

out, err = process.communicate()
if process.returncode == 0:
    print(line)
else:
    print("====")
    print(process.stderr)
    print(out)
    raise Exception(str(err))
