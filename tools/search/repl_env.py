
import sys
import os


# Add the client folder to sys.path
CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "fusion360gym", "client")
if CLIENT_DIR not in sys.path:
    sys.path.append(CLIENT_DIR)

from gym_env import GymEnv


class ReplEnv(GymEnv):

    def set_target(self, target_file):
        """Setup search and connect to the Fusion Gym"""
        # Set the target
        r = self.client.set_target(target_file)
        self.check_response("set_target", r)
        response_json = r.json()
        if "data" not in response_json or "graph" not in response_json["data"]:
            raise Exception("[set_target] response graph missing")
        return (response_json["data"]["graph"],
                response_json["data"]["bounding_box"])

    def revert_to_target(self):
        """Revert to the target to start the search again"""
        r = self.client.revert_to_target()
        self.check_response("revert_to_target", r)
        response_json = r.json()
        if "data" not in response_json or "graph" not in response_json["data"]:
            raise Exception("[revert_to_target] response graph missing")
        return response_json["data"]["graph"]

    def get_empty_graph(self):
        """Get an empty graph to kick things off"""
        return {
            "directed": False,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": []
        }
 
    def boundary_points(self, min_point, max_point):
        """Return points needed to make a boundary extrude encompassing model"""
        start_face = []
        end_face = []
        start_face.append(min_point)
        start_face.append((min_point[0], min_point[1], max_point[2]))
        start_face.append((min_point[0], max_point[1], max_point[2]))
        start_face.append((min_point[0], max_point[1], min_point[2]))

        end_face.append((max_point[0], min_point[1], min_point[2]))
        end_face.append((max_point[0], min_point[1], max_point[2]))
        end_face.append((max_point[0], max_point[1], max_point[2]))
        end_face.append((max_point[0], max_point[1], min_point[2]))

        return start_face, end_face

    def get_bounding_graph(self, min_point, max_point):
        """Wrapper to extrude a bounding rectangle around a solid model"""
        _min_point = min_point
        _max_point = max_point

        start_face, end_face = self.boundary_points(_min_point, _max_point)
        distance = _max_point[0] - _min_point[0]

        # Create a new sketch vias the server client
        r = self.client.add_sketch("XY")
        response_json = r.json()
        sketch_name = response_json["data"]["sketch_name"]

        # Add points to new sketch
        for i, point in enumerate(start_face):
            r = self.client.add_point(sketch_name, 
                {"x": point[0], "y": point[1], "z":point[2]})
            response_json = r.json()

        # Close profile of points to make a face to extrude
        r = self.client.close_profile(sketch_name)
        response_json = r.json()
        profile_id = (list(response_json['data']['profiles'].keys())[0])

        # Extrude by distance between start and end face
        r = self.client.add_extrude(sketch_name, profile_id, -1 * distance, "NewBodyFeatureOperation")

        # Convert extrusion to graph and return graph format
        r = self.client.graph(file='', dir='', format='PerFace')
        response_json = r.json()
        return response_json['data']['graph']

    def extrude(self, start_face, end_face, operation):
        """Extrude wrapper around the gym client"""
        is_invalid = False
        return_graph = None
        return_iou = None
        r = self.client.add_extrude_by_target_face(
            start_face, end_face, operation)
        if r is not None and r.status_code == 200:
            response_json = r.json()
            if ("data" in response_json and
                    "graph" in response_json["data"] and
                    "iou" in response_json["data"]):
                return_graph = response_json["data"]["graph"]
                return_iou = response_json["data"]["iou"]
        return return_graph, return_iou

    def extrudes(self, actions, revert=False):
        """Extrudes wrapper around the gym client"""
        if len(actions) == 0:
            return None, None
        is_invalid = False
        return_graph = None
        return_iou = None
        r = self.client.add_extrudes_by_target_face(actions, revert)
        if r is not None and r.status_code == 200:
            response_json = r.json()
            if ("data" in response_json and
                    "graph" in response_json["data"] and
                    "iou" in response_json["data"]):
                return_graph = response_json["data"]["graph"]
                return_iou = response_json["data"]["iou"]
        return return_graph, return_iou

    def screenshot(self, file):
        """Save out a screenshot"""
        r = self.client.screenshot(file)
        return r is not None and r.status_code == 200
