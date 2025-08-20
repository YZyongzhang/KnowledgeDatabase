import random
import pickle
class GraphStruct:
    def __init__(self):
        self.nodes = list()
        self.pins = list()
        self.block = None # 目前将block设置为1块。
    def load(self , path:str):
        assert len(self.nodes) == 0
        assert len(self.pins) == 0
        assert self.block == None
        with open(path , 'rb') as f:
            graphstruct = pickle.load(f)
        self.nodes = graphstruct.nodes
        self.pins = graphstruct.pins
        self.block = graphstruct.block
    

class Graph:
    def __init__(self , database_path=None):
        """
        可以输入本地的数据库路径进行load
        """
        
        self.graph_struct = GraphStruct()
        if database_path is not None:
            self.graph_struct.load(database_path)
        self.pins = self.graph_struct.pins
        

    def reseive_node(self , node):
        pin_id = self.rule_base_getid(node)

        if len(self.pins) != 0:
            pin = self.pins[pin_id]
            node.pre = pin
        else:
            node.pre = None
        self.pins.append(node.next)

        assert self.graph_struct != None
        self.graph_struct.nodes.append(node)



    def rule_base_getid(self , node):
        return 0
    

    def save(self , path):
        if path is not None:
            save_path = f"{path}/database.pkl"
        else:
            save_path = f"database.pkl"
        with open(save_path , 'wb') as f:
            pickle.dump(self.graph_struct , f)


