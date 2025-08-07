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