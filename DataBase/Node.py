class Node:
    def __init__(self):
        self.key = None
        self.value = None
        
        self.pre = None
        self.next = None
    def push(self , data):
        """
        只获取输入json数据data的第一个key和第一个key对应的value
        """
        assert self.key == None 
        assert self.value == None
        key = list(data.keys())[0] # 取出第一个key。
        self.key = key
        value = data[key]
        self.value = value

    def clear(self):
        self.key = None
        self.value = None