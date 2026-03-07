
class UploadedFileWrapper:
    def __init__(self, path, name, content, createdAt):
        self.name = name
        self.path = path
        self.content = content
        self.createdAt = createdAt

    def read(self):
        with open(self.path, "rb") as f:
            return f.read()
    
    def to_dict(self):
        return {
            'name': self.name,
            'path': self.path,
            'content': self.content,
            'createdAt': self.createdAt
        }