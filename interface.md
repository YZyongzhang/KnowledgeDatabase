# DataBase Knowledga
### interface
---
```
1. input(data)
    def input(self , data ):
        """
        data : {
            "key":"value"
        }
        """
    输入json格式的data。一般是将知识进行总结成key，知识内容写为value。该接口将输入的data写入到数据库中去。

2. get()
    def get(self)：
    从数据库中获取到所有的知识

3. regex_key(key)
    def regex_key(self,key):
    
    输入的是数据库的key，模糊返回匹配的所有的key-value对

```