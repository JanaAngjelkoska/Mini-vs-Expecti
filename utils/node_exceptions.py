
class NoChildrenCalculatedYetException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return (f'No children found in node with state \n {self.message} \n'
                f'Call calculate_children() to fetch them')
