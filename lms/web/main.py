import argparse
import os
import pathlib
import signal
import subprocess
import sys

home = os.path.expanduser("~")


def until_started(log_path):
    cmd = f'tail -F {log_path}'
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    for line in iter(proc.stdout.readline, b''):
        line = str(line, 'utf-8')
        if line == 'started\n':
            print("successful to start lms_web")
            proc.terminate()
            return True
        print(line, end='')
    return False


def main():
    parser = argparse.ArgumentParser(prog='lms_web',
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

    #  lms_web start --host xx --port xx
    parser_start = subparsers.add_parser('start', help='start help')
    parser_start.add_argument('--host', type=str, default="0.0.0.0", help='global_params help')
    parser_start.add_argument('--port', type=int, default=18080, help='global_params help')
    parser_start.add_argument('--workers', type=int, default=3, help='global_params help')
    parser_start.add_argument('--database', type=str, default=home + '/.lms/lms_web.db', help='global_params help')
    parser_start.add_argument('--logfile', type=str, help='global_params help')
    parser_start.add_argument('--loglevel', type=str, default='INFO', choices=['ERROR', 'WARN', 'INFO', 'DEBUG'],
                              help='Log level')

    parser_stop = subparsers.add_parser('stop', help='stop help')
    parser_stop.add_argument('--database', type=str, default=home + '/.lms/lms_web.db', help='global_params help')
    parser_stop.add_argument('--force', default=False, action='store_true', help='global_params help')

    args = parser.parse_args()
    import os
    pid_path = f'{os.path.expanduser("~")}/.lms/lms_web.pid'
    if args.sub_command == "start":
        if os.path.isfile(pid_path):
            print("The lms_web is already started")
            sys.exit(1)

        log_path = f'{os.path.expanduser("~")}/.lms/logs/lms_web.log'
        os.makedirs(pathlib.Path(log_path).parent, exist_ok=True)

        cmd = f"{sys.executable} -u -m lms.web.launcher {' '.join(sys.argv[2:])} >{log_path} 2>&1 "
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                preexec_fn=lambda: os.setpgrp())
        if until_started(log_path):
            with open(pid_path, "w") as f:
                f.write(str(proc.pid))
        else:
            if proc.poll() is not None:
                print(f"failed to start lms_web with exit code:{proc.returncode}")
                sys.exit(1)

    if args.sub_command == "stop":
        if not args.force:
            try:
                os.environ['database'] = args.database
                from lms.web.utils import peewee_repo
                from lms.web.utils.peewee_repo import db
                db.connect(reuse_if_open=True)
                deployed_models = peewee_repo.query_model_list(status='deployed')
                if len(deployed_models) > 0:
                    raise Exception("There are deployed models in this cluster. Please undeployment before stop")
            finally:
                if not db.is_closed():
                    db.close()

        if os.path.isfile(pid_path):
            with open(pid_path, "r") as f:
                pid = f.read()

            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            os.remove(pid_path)
            print("successful to stop lms_web")
        else:
            print("The lms_web isn't started")
            sys.exit(1)


if __name__ == "__main__":
    main()
