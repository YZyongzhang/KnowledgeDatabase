class Node:
    def __init__(self):
        self.key = None
        self.value = None
        
        self.pre = None
        self.next = None
    def push(self):
        assert self.key != None 
        assert self.value != None
