


class BusinessException(Exception):
    def __init__(self, code:int,msg:str):
        self.code = code
        self.msg = msg
