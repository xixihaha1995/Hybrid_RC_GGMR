def recv_send(client_msg):
    return [client_msg + ", Echo: Hello Matlab client", "second result"]
res = recv_send(client_msg)