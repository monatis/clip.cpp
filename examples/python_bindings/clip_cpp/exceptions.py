

class RepositoryNotFoundError(Exception):
    '''Raise when Provided repo_id is not Found'''

    def __init__(self, *args):
        message = """
        the Provided repo_id is not Found or maybe it's private repo which is not implemented yet.
        """
        super(RepositoryNotFoundError, self).__init__(message, *args)


class RepositoryFileNameNotFound(Exception):
    '''Raise when a Hugging Face file is Not Found'''

    def __init__(self,  *args):
        message = """
        Hugging Face file in the Provided repo_id is not Found.
        """
        super(RepositoryFileNameNotFound, self).__init__(message, *args)


class FileNameAlreadyExists(Exception):
    '''Raise when a Hugging Face file is exists or downloaded'''

    def __init__(self,  *args):
        message = """
        Hugging Face file is already exists or downloaded and it's size match the size of that in the network.
        """
        super(FileNameAlreadyExists, self).__init__(message, *args)
