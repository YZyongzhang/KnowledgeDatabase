import sys
sys.path.append(r"C:\Users\31120\Desktop\KnowledgeDatabase")
import json
import re
from DataBase.Node import Node
from DataBase.Graph import Graph
import os
import copy
class DataImpl:
    """
    path: database path , this file is a json
    """
    def __init__(self , graph = None):
        self.graph = graph
        self.node = Node()
    def _input(self , data):
        
        
        self.node.push(data)
        copy_node = copy.deepcopy(self.node) # 每一次copy都相当于丢掉了copy的上一次引用，将上一次引用放到丢到内存中。如果程序一直挂起。后期可能会造成内存泄漏。可能想办法把这个凝结成一块
        self.graph.reseive_node(copy_node) # 这里犯了一个错误，这里执行了self.clear但是这个地方是引用数据类型，所以这里的话放到列表里面的数据直接就被clear掉了。这里需要用一个deepcopy
        self.node.clear()

    def input(self, data):
        """
        data : json , input to the database
        """

        self._input(data)

    def _get_all(self):
        """
        利用图的遍历进行获取到所有的node节点
        """
        pass
    
    def get_all(self):
        """
        get the database
        """
        return self._get_all()
    

    def get_node(self , key):
        """
        模糊遍历每一个node的key，返回匹配的value
        """
        pass
    
    def search(self,pattern):
        pass
    
    def regex_keys(self , key):
        """
        key : josn key 
        """
        # pattern = f"*{key}*"
        pattern = re.compile(f".*{re.escape(key)}.*")
        return {k: v for k, v in self._get().items() if re.search(pattern, k)}
    
    def file_write(self , file_path):
        """
        给定file_path的text文件，将该文件转化为json
        """
        pass

    def load_json_window(self):
        """
        创建一个可供用户写入知识的窗口。
        用户在该窗口写入的内容将会把load在数据库中。
        """
        pass

    def save(self , path = None):
        if path is not None:
            os.makedirs(path , exist_ok=True)
        self.graph.save(path)

graph = Graph('./database.pkl')
data_impl = DataImpl(graph=graph)