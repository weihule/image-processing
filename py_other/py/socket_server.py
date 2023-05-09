import socket
import threading


def main():
    # AF_INET 表示ipv4
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # socket s 用于监听；socket c用于和客户端通信
        s.bind(("0.0.0.0", 1234))
        s.listen()
        
        c, addr = s.accept()
        with c:
            print(addr, "connected")

            while(True):
                # 1024 代表一次性接受数据的最大长度
                datas = c.recv(1024)
                if not datas:
                    break
                c.sendall(datas)


def handle_client(c, addr):
    print(addr, "connected")

    while True:
        datas = c.recv(1024)
        if not datas:
            break
        c.sendall(datas)

def main2():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 1234))
        s.listen()

        while True:
            c, addr = s.accept()
            t = threading.Thread(target=handle_client, args=(c, addr))
            t.start()


if __name__ == "__main__":
    main2()

