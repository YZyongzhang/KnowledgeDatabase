import json
import re
class Data_impl:
    """
    path: database path , this file is a json
    """
    def __init__(self , path):
        self.path = path
    
    def _input(self , data):
        with open(self.path , 'a') as f:
            json.dump(data , f)
    
    def input(self, data):
        """
        data : json , input to the database
        """
        self._input(data = data)

    def _get(self):
        with open(self.path , 'r') as f:
            data = json.load(f)
        return data
    
    def get(self):
        """
        get the database
        """
        return self._get()
    
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
    