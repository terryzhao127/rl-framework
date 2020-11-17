from fabric import Connection
c = Connection(
    host="localhost",
    connect_kwargs={
        "password": "dongdong888",
    },
)
result = c.run('uname -s', hide=True)
msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
print(msg.format(result))