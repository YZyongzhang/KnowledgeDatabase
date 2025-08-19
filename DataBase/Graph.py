import random

class Graph:
    def __init__(self):
        
        self.pins = None # graph中的每一块暴露的引脚，用来接受新的node
    def reseive_node(self , node):
        pin_id = self.rule_base_getid(node)
        pin = random.choice(self.pins[pin_id])
        node.pre = pin
        self.pins.remove(pin)
        self.pins.appent(node.next)
        assert self.graph != None
        self.graph.add(node)

    def load_graph(self , database_path):
        database_graph = self.load_all_database(database_path)
        self.graph = database_graph
        self.pins = self.graph.pins # 这里存储数据的时候要注意不要遇到死循环。或者用递归的方式获取？


    def rule_base_getid(self , node):
        pass

    def load_all_database(self, database_path):
        pass

    def add(self, node):
        pass