import yaml
    
DATAPATH = './DataBase/raw_files/DataBase.yaml'

class DataBase:
    def __init__(self):
        self.datapath = DATAPATH
        self.databaseimple = DataBaseImple
        self.total_keys = self.sort_total()
        self.knowledge_dict = self.databaseimple._sort_knowledge() # knowledge_tree

    def sort_total(self):
        total_key = self.get_key(self.knowledge_dict)
        return total_key
    
    def search(self, search_content):
        # 这个就是查找算法。将整个数据库想象成一个树，那么查找一个节点就可以用深度(广度)优先遍历算法
        # 这就要求我们的database必须要有一个根节点
        if search_content in self.total_keys: # 先判断是否在整个树内，因为我最开始就使用了一次递归判断了一下
            # 我的需求是存起来每一次遍历的路径，然后将遍历成功的路径输出出来
            self.search_tree(search_content)

    def search_tree(self,node):
        # 在我数据库存储结构中，由于宽度比较高，深度比较小，因此更适合使用广度优先遍历
        # 因为我这个是一个字典，没有用树的结构进行存储，所以这个地方只是使用了遍历的思路
        traces = list()
        trace = list()
        son_node = self.son_node(self.knowledge_dict)
        for k , v in self.knowledge_dict:
            if node != k:
                current_dict = self.knowledge_dict[k]
                son_nodes = self.son_node(current_dict)
                trace.append(k)
                for son_node in son_nodes:
                    pass


    def son_node(self, dict):
        key = list()
        for k , v in dict.item():
            key.append(k)

        return key


    def get_key(self,dict):
        # 递归算法
        key = set()
        for key_i , value in dict.items():
            key.append(key_i)
            if isinstance(value , dict):
                key.update(self.get_key(value))
        return key

class DataBaseImple:
    @classmethod
    def _get_DataBase(cls):
        with open(DATAPATH , 'rb') as f:
            Data_Dict = yaml.load( f , Loader=yaml.FullLoader)

        return Data_Dict
    
    @classmethod
    def _search_knowledge(cls,knowledge_dict):

       
        while True:
            frist_level_knowledge_key  = input("which field do you want to search?:")
            if frist_level_knowledge_key in knowledge_dict.keys():
                geted_content = knowledge_dict[frist_level_knowledge_key]

            elif frist_level_knowledge_key == 'q':
                break

            else:
                print("search key error , that not in key database? you can input key again or input q to quit:")
                frist_level_knowledge_key  = input("you input here:")

        return geted_content

        

    @classmethod
    def _show_knowledge(cls,geted_content):
        pass
    @classmethod
    def _sort_knowledge(cls):
        knowledge_dict = DataBase._get_DataBase()
        return knowledge_dict

