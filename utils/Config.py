# -*- coding: utf-8 -*-

from ConfigParser import SafeConfigParser
import json
import os
import sys

class Config(object):
    def __init__(self, path_config=None):
        self.parser = SafeConfigParser()
        self.parser.read("./config/path.ini")
        if path_config is not None:
            if not os.path.exists(path_config):
                print "Error!: path_config=%s does not exist." % path_config
                sys.exit(-1)
            self.parser.read(path_config)

    def getpath(self, key):
        return self.str2None(json.loads(self.parser.get("path", key)))

    def getint(self, key):
        return self.parser.getint("hyperparams", key)

    def getfloat(self, key):
        return self.parser.getfloat("hyperparams", key)

    def getbool(self, key):
        return self.parser.getboolean("hyperparams", key)

    def getstr(self, key):
        return self.str2None(json.loads(self.parser.get("hyperparams", key)))

    def getlist(self, key):
        xs = json.loads(self.parser.get("hyperparams", key))
        xs = [self.str2None(x) for x in xs]
        return xs

    def getdict(self, key):
        xs  = json.loads(self.parser.get("hyperparams", key))
        for key in xs.keys():
            value = self.str2None(xs[key])
            xs[key] = value
        return xs

    def str2None(self, s):
        if (isinstance(s, unicode) or isinstance(s, str)) and s == "None":
            return None
        else:
            return s


