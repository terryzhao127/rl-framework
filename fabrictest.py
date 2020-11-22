# fabrictest.py
from fabric import Connection
c = Connection(
    host="172.17.0.2",
    user="root",
    connect_kwargs={
        "password": "123456",
    },
)
result = c.run('uname -s', hide=True)
msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
print(msg.format(result))

result = c.run('cd /workplace/rl-framework/ && ~/miniconda3/envs/framework/bin/pytest')
print('pytest finish')
msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
print(msg.format(result))
