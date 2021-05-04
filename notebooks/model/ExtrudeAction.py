from dataclasses import dataclass
from uuid import UUID, uuid4

import numpy as np

from .Operation import Operation

@dataclass(frozen=True)
class ExtrudeAction:
    # UUID of start face in the target
    start_face: str
    # UUID of end face in the target
    end_face: str
    # the type of operation (defined above)
    operation: Operation

    def encode(self) -> np.array:
        """ Encodes an action to tf-agent compatible format"""
        # First, try converting to ids UUID (as stated)
        try:
            start_face_uuid = UUID(self.start_face)
            end_face_uuid = UUID(self.end_face)            
            start_lower_64 = start_face_uuid.int & ((2 ** 64) - 1)
            start_upper_64 = start_face_uuid.int >> 64 & ((2 ** 64) - 1)
            end_lower_64 = end_face_uuid.int & ((2 ** 64) - 1)
            end_upper_64 = end_face_uuid.int >> 64 & ((2 ** 64) - 1)
            
            return np.array([
                start_upper_64,
                start_lower_64,
                end_upper_64,
                end_lower_64,
                int(self.operation)
            ], dtype=np.uint64)
        except ValueError:
            # Since not all ids appear to be UUIDs, if this doesn't 
            # work, try using just an int
            return np.array([
                0,
                int(self.start_face),
                0,
                int(self.end_face),
                int(self.operation)
            ], dtype=np.uint64)
            

    def decode(encoded: np.array):
        if(encoded[0] == 0) and (encoded[2] == 0):
            # this was not a proper UUID, just decode the int
            return ExtrudeAction(
                str(encoded[1]),
                str(encoded[3]),
                Operation(encoded[4])
            )
        else:
            # a proper UUID was encoded
            start_face = (int(encoded[0]) << 64) | encoded[1]
            end_face   = (int(encoded[2]) << 64) | encoded[3]
            return ExtrudeAction(
                str(UUID(int=start_face)),
                str(UUID(int=end_face)),
                Operation(encoded[4])
            )
            
