from dataclasses import dataclass
from uuid import UUID, uuid4

import numpy as np

from .Operation import Operation

@dataclass(frozen=True)
class ExtrudeAction:
    # UUID of start face in the target
    start_face: uuid4
    # UUID of end face in the target
    end_face: uuid4
    # the type of operation (defined above)
    operation: Operation

    def encode(self) -> np.array:
        """ Encodes an action to tf-agent compatible format"""
        start_lower_64 = self.start_face.int & ((2 ** 64) - 1)
        start_upper_64 = self.start_face.int >> 64 & ((2 ** 64) - 1)
        end_lower_64 = self.end_face.int & ((2 ** 64) - 1)
        end_upper_64 = self.end_face.int >> 64 & ((2 ** 64) - 1)

        return np.array([
            start_upper_64,
            start_lower_64,
            end_upper_64,
            end_lower_64,
            int(self.operation)
        ], dtype=np.uint64)

    def decode(encoded: np.array):
        start_face = (int(encoded[0]) << 64) | encoded[1]
        end_face   = (int(encoded[2]) << 64) | encoded[3]
        return ExtrudeAction(
            UUID(int=start_face),
            UUID(int=end_face),
            Operation(encoded[4])
        )