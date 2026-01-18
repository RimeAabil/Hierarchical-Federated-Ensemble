import socket
import struct
import pickle

def send_msg(sock, msg):
    """
    prefixes each message with a 4-byte length (network byte order)
    """
    msg = pickle.dumps(msg)
    sock.sendall(struct.pack('>I', len(msg)) + msg)

def recv_msg(sock):
    """
    Read message length and unpack it into an integer
    """
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return pickle.loads(recvall(sock, msglen))

def recvall(sock, n):
    """
    Helper function to recv n bytes or return None if EOF is hit
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
