import socket


def main():
    # AF_INET 表示ipv4
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("127.0.0.1", 1234))
        s.sendall(b"Hello Whl")
        datas = s.recv(1024)
        print("Receives: ", repr(datas))
        

if __name__ == "__main__":
    main()

