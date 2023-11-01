import argparse

# create the top-level parser
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--foo', action='store_true', help='foo help')
subparsers = parser.add_subparsers(help='sub-command help')


# create the parser for the "a" command
parser_a = subparsers.add_parser('a', help='a help')

# parser_a.add_argument('model_path',type=str)

subsub = parser_a.add_subparsers(help="sub-sub help")

parser_a.add_argument()

parser_a1 = subsub.add_parser("a1")
parser_a1.add_argument("--pp", type=int, help="pp help", required=True)
parser_a2 = subsub.add_parser("a2")
parser_a2.add_argument("--tt", type=int, help="tt help", required=False)

parser_a.add_argument('model_path',type=str)

# create the parser for the "b" command
parser_b = subparsers.add_parser('b', help='b help')
parser_b.add_argument('--baz', choices='XYZ', help='baz help')

parser.parse_args()
print("Done")