from fabric import Connection
c = Connection(
    host="localhost",
    connect_kwargs={
        "password": "123456",
    },
)
result = c.run('uname -s', hide=True)
msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
print(msg.format(result))
result = c.run("conda activate framework", hide=True)
result = c.run('pytest -s /workplace/rl-framework/leatest.py', hide=True)
msg = "got stdout:\n{0.stdout}"
print(msg.format(result))
