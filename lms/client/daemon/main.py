import argparse

from lms.client.daemon.serving import make_app


def main():
    parser = argparse.ArgumentParser(prog='lmsd',
                                     description="""
                                        The lmsd is daemon server of lms. It runs on the node of the lms client, 
                                        exposing metrics——gpu, cpu, number of request etc.. 
                                         """)

    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1.0"
    )

    parser.add_argument('--host', type=str, default="0.0.0.0", help='global_params help')
    parser.add_argument('--port', type=int, default="8082", help='global_params help')
    parser.add_argument('--interval', type=int, default=15, help='global_params help')

    args = parser.parse_args()
    make_app(args.interval).run(host='0.0.0.0', port=args.port, single_process=True)


if __name__ == "__main__":
    main()
