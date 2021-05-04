from enum import IntEnum

class Operation(IntEnum):
    JoinFeatureOperation = 1
    CutFeatureOperation = 2
    IntersectFeatureOperation = 3
    NewBodyFeatureOperation = 4
    
    def from_string(string: str):
        if string == "JoinFeatureOperation":
            return Operation.JoinFeatureOperation
        elif string == "CutFeatureOperation":
            return Operation.CutFeatureOperation
        elif string == "IntersectFeatureOperation":
            return Operation.IntersectFeatureOperation
        elif string == "NewBodyFeatureOperation":
            return Operation.NewBodyFeatureOperation
        else:
            raise ValueError(f"Unknown operation: {string}.")
            
    def to_string(self) -> str:
        if self == Operation.JoinFeatureOperation:
            return "JoinFeatureOperation"
        elif self == Operation.CutFeatureOperation:
            return "CutFeatureOperation"
        elif self == Operation.IntersectFeatureOperation:
            return "IntersectFeatureOperation"
        elif self == Operation.NewBodyFeatureOperation:
            return "NewBodyFeatureOperation"
