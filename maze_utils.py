
import random
import numpy as np
from PIL import Image
from typing import Tuple, List


# DiPPeR uses 100x100 maps, 0.1 m/pixel, so world range is [-5, 5] meters
MAP_SIZE = 100
RESOLUTION = 0.1  # m per pixel
ORIGIN = -5.0     # meters (map center at 0,0 in world coords)


class DisjointSet:
    def __init__(self, nodes):
        self.node_mapping = {}
        for i, val in enumerate(nodes):
            self.node_mapping[val] = self._DSNode(val, i)

    def find(self, node):
        return self.find_node(node).parent

    def find_node(self, node):
        n = self.node_mapping[node]
        if isinstance(n.parent, int):
            return n  # root node
        parent_node = self.find_node(n.parent.val)
        n.parent = parent_node
        return parent_node

    def union(self, node1, node2):
        p1, p2 = self.find_node(node1), self.find_node(node2)
        if p1.parent != p2.parent:
            p1.parent = p2

    class _DSNode:
        def __init__(self, val, parent):
            self.val = val
            self.parent = parent


def generate_kruskal_maze(graph_size: int = 9, cell_thickness: int = 10,
                          wall_thickness: int = 1, seed: int = None) -> np.ndarray:
    """
    Generate a random maze using Kruskal's algorithm.
    Returns 100x100 uint8 image: 255=free, 0=obstacle.
    """
    if seed is not None:
        random.seed(seed)

    nodes = [(i, j) for j in range(graph_size) for i in range(graph_size)]
    neighbors = lambda n: [
        (n[0] + dx, n[1] + dy)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= n[0] + dx < graph_size and 0 <= n[1] + dy < graph_size
    ]

    edges = [(n, nb) for n in nodes for nb in neighbors(n)]
    maze_edges = []
    ds = DisjointSet(nodes)

    while len(maze_edges) < len(nodes) - 1:
        edge = edges.pop(random.randint(0, len(edges) - 1))
        if ds.find(edge[0]) != ds.find(edge[1]):
            ds.union(edge[0], edge[1])
            maze_edges.append(edge)

    w = graph_size * (cell_thickness + wall_thickness) + wall_thickness
    img = np.zeros((w, w), dtype=np.uint8)

    for edge in maze_edges:
        min_x = wall_thickness + min(edge[0][0], edge[1][0]) * (cell_thickness + wall_thickness)
        max_x = wall_thickness + max(edge[0][0], edge[1][0]) * (cell_thickness + wall_thickness)
        min_y = wall_thickness + min(edge[0][1], edge[1][1]) * (cell_thickness + wall_thickness)
        max_y = wall_thickness + max(edge[0][1], edge[1][1]) * (cell_thickness + wall_thickness)
        img[min_x:max_x + cell_thickness, min_y:max_y + cell_thickness] = 255

    padding = MAP_SIZE - w
    if padding > 0:
        pad_l = padding // 2
        pad_r = padding - pad_l
        img = np.pad(img, ((pad_l, pad_r), (pad_l, pad_r)), constant_values=255)

    return img


def create_simple_corridor_maze() -> np.ndarray:
    """Simple horizontal corridor for easy debugging."""
    img = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    img[40:60, :] = 255  # horizontal corridor
    return img


def create_l_shaped_maze() -> np.ndarray:
    """L-shaped free space."""
    img = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    img[20:80, 20:40] = 255   # vertical leg
    img[40:80, 20:80] = 255   # horizontal leg
    return img


def create_room_maze() -> np.ndarray:
    """Single room with obstacles."""
    img = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.uint8) * 255
    img[10:20, :] = 0
    img[80:90, :] = 0
    img[:, 10:20] = 0
    img[:, 80:90] = 0
    # inner obstacles
    img[40:50, 30:70] = 0
    img[60:70, 30:70] = 0
    return img


def maze_to_model_input(img: np.ndarray) -> np.ndarray:
    """
    Convert maze image to model input format: (H, W, 3) float32 in [0, 1].
    Model expects white=free (1), black=obstacle (0).
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    return img


def world_to_pixel(x: float, y: float) -> Tuple[int, int]:
    """Convert world coords (meters, -5..5) to pixel indices. Origin at bottom-left."""
    px = int((x - ORIGIN) / RESOLUTION)
    py = int((MAP_SIZE - 1) - (y - ORIGIN) / RESOLUTION)  # row 0 = top = y=5
    return int(np.clip(px, 0, MAP_SIZE - 1)), int(np.clip(py, 0, MAP_SIZE - 1))


def pixel_to_world(px: int, py: int) -> Tuple[float, float]:
    """Convert pixel indices to world coords."""
    x = px * RESOLUTION + ORIGIN
    y = (MAP_SIZE - 1 - py) * RESOLUTION + ORIGIN
    return x, y


def is_cell_free(occupancy: np.ndarray, px: int, py: int, free_threshold: float = 0.5) -> bool:
    """Check if pixel (px, py) is free. occupancy: 0=obstacle, 255 or 1=free."""
    if occupancy.max() > 1:
        val = occupancy[py, px] / 255.0
    else:
        val = occupancy[py, px]
    return val >= free_threshold


def compute_path_metrics(path: np.ndarray, occupancy: np.ndarray,
                         free_threshold: float = 0.5, n_samples: int = 50) -> dict:
    """
    path: (N, 2) in world coords
    occupancy: (H, W) or (H, W, 3), 255 or 1 = free
    """
    if occupancy.ndim == 3:
        occ = occupancy[:, :, 0] if occupancy.shape[2] >= 1 else occupancy[:, :, 0]
    else:
        occ = occupancy

    # Path length (sum of segment lengths)
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(np.sum(segment_lengths))

    # Feasibility: sample points along path and check collisions
    collisions = 0
    for i in range(len(path) - 1):
        for t in np.linspace(0, 1, n_samples):
            pt = path[i] + t * (path[i + 1] - path[i])
            px, py = world_to_pixel(pt[0], pt[1])
            px, py = int(px), int(py)
            if 0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE:
                if not is_cell_free(occ, px, py, free_threshold):
                    collisions += 1
                    break

    feasible = collisions == 0
    collision_rate = collisions / max(1, len(path) - 1)

    return {
        'path_length_m': path_length,
        'n_waypoints': len(path),
        'feasible': feasible,
        'collision_rate': collision_rate,
    }
