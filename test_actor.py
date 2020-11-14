# actor part test: communication
import subprocess
import pytest
import zmq
from dqn.protobuf.data import Data, bytes2arr

def test_con():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.17.0.3:7281")

    subp1 = subprocess.Popen('python3 actor.py', shell = True)
    weight = b''
    socket.send(weight)
    rec = socket.recv()

    assert rec

def test_dataFormat():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.17.0.3:7281")

    subp1 = subprocess.Popen('python3 actor.py', shell=True)
    weight = b''
    socket.send(weight)
    data = Data()
    data.ParseFromString(socket.recv())

    assert isinstance(data.state,bytes) and isinstance(data.action,int) and isinstance(data.reward,float) and \
           isinstance(data.next_state, bytes) and isinstance(data.done, bool) and isinstance(data.epoch, int)
