import os
import json
import datetime

class Config:
    def __init__(self, parser, agent) -> None:
        self.argdic = vars(parser.parse_args())
        self.agentdic = agent.export_config()

    def save_config(self, path, type):
        
        assert isinstance(path, str)
        path = os.path.expanduser(path)
        os.makedirs(os.path.expanduser(path), exist_ok=True)

        argjs = json.dumps(self.argdic)
        agentjs = json.dumps(self.agentdic)
        cur_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(cur_time,'%m-%d-%H-%M')
        if type == "actor":
            file = open(path + time_str + 'learner_config.json', 'w')
        elif type == "learner":
            file = open(path + time_str + 'learner_config.json', 'w')
        file.write(argjs)
        file.write(agentjs)
