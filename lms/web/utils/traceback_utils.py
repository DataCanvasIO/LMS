import traceback


def get_tarceback(type, value, tb):
    return '\n'.join(str(x) for x in traceback.format_exception(type, value, tb))
