import socket
from contextlib import closing


def is_port_available(port):
    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Try binding the socket to the specified port
        sock.bind(('localhost', port))
        return True
    except socket.error:
        # Port is in use
        return False
    finally:
        # Close the socket
        sock.close()


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]