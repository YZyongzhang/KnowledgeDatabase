import yaml
path = './tt.yaml'
with open(path , 'rb') as f:
    tec = yaml.load(f , Loader=yaml.FullLoader)
    
DATAPATH = './DataBase/DataBase.yaml'

class DataBase:
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

        

    @classmethod
    def _show_knowledge(cls,geted_content):
        pass
    @classmethod
    def _sort_knowledge(cls):
        knowledge_dict = DataBase._get_DataBase()

