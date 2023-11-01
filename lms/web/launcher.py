import argparse
import logging
import os
from configparser import ConfigParser
from pathlib import Path
from secrets import token_urlsafe

home = os.path.expanduser("~")


def launch(args):
    if args.logfile is not None:
        logs_dir = Path(args.logfile).parent
        if not logs_dir.exists():
            import os
            os.makedirs(logs_dir, exist_ok=True)
        logging.basicConfig(filename=args.logfile, level=logging.getLevelName(args.loglevel))
    import os
    os.environ['database'] = args.database
    os.makedirs(os.path.dirname(os.environ['database']), exist_ok=True)

    config_path = f'{os.path.expanduser("~")}/.lms/lms_web.config'
    if os.path.exists(config_path):
        config = ConfigParser()
        config.read(config_path)
        if not config.__contains__('web') or not config['web'].__contains__('server_id'):
            raise Exception("There is no 'web.id' in the file:lms_web.config")
        else:
            server_id = config['web']['server_id']
    else:
        server_id = token_urlsafe(16)
        config = ConfigParser()
        config.add_section('web')  # 添加table section
        config.set('web', 'server_id', server_id)
        with open(config_path, 'w', encoding='utf-8') as file:
            config.write(file)

    from lms.web.utils.peewee_repo import create_database_tables
    create_database_tables()
    from lms.web.route import make_app

    make_app(server_id).run(host=args.host, port=args.port, single_process=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help='global_params help')
    parser.add_argument('--port', type=int, default=18080, help='global_params help')
    parser.add_argument('--workers', type=int, default=3, help='global_params help')
    parser.add_argument('--database', type=str, default=home + '/.lms/lms_web.db', help='global_params help')
    parser.add_argument('--logfile', type=str, help='global_params help')
    parser.add_argument('--loglevel', type=str, default='INFO', choices=['ERROR', 'WARN', 'INFO', 'DEBUG'],
                        help='Log level')
    args = parser.parse_args()
    launch(args)
