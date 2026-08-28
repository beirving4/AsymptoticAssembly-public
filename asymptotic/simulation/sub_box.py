from __future__ import annotations

import numpy as np, pdb

from time import time
from attrs import define, field

from ..utils.jackknife import get_jackknife_subbox_samples


CoordinateType = tuple[float, float, float] | np.ndarray
MaskType = bool | np.ndarray


@define(slots=True)
class PopulationProperties:
    particle_ids: np.ndarray = field(default=np.empty(0, dtype=int))
    particle_indices: np.ndarray = field(default=np.empty(0, dtype=int))

    group_ids: np.ndarray = field(default=np.empty(0, dtype=int))
    group_indices: np.ndarray = field(default=np.empty(0, dtype=int))

    subhalo_ids: np.ndarray = field(default=np.empty(0, dtype=int))
    subhalo_indices: np.ndarray = field(default=np.empty(0, dtype=int))

    @property
    def num_particles(self) -> int:
        return len(self.particle_ids)
    
    @property
    def num_groups(self) -> int:
        return len(self.group_ids)
    
    @property
    def num_subhalos(self) -> int:
        return len(self.subhalo_ids)


@define(slots=True)
class SubBox:
    size: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]

    props: PopulationProperties | None = field(default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        self.props = PopulationProperties() if self.props is None else self.props

    def __repr__(self) -> str:
        return (
            f"SubBox(L={self.size} cMpc/h, x_range={self.x_range}, "
            f"y_range={self.y_range}, z_range={self.z_range}, "
            f"num_groups ={self.num_groups}, "
            f"num_subhalos={self.num_subhalos})"
        )
    def is_in_box(self, points: CoordinateType) -> MaskType:
        if isinstance(points, tuple) and len(points) == 3:
            x, y, z = points
            return (
                self.x_range[0] <= x <= self.x_range[1] and 
                self.y_range[0] <= y <= self.y_range[1] and 
                self.z_range[0] <= z <= self.z_range[1]
            )
        else:
            x, y, z = points.T
            return (
                (self.x_range[0] <= x) & (x <= self.x_range[1]) &
                (self.y_range[0] <= y) & (y <= self.y_range[1]) &
                (self.z_range[0] <= z) & (z <= self.z_range[1])
            )
        
    @property
    def volume(self) -> float:
        return self.size**3

    @property
    def num_particles(self) -> int:
        return self.props.num_particles if self.props is not None else 0

    @property
    def num_groups(self) -> int:
        return self.props.num_groups if self.props is not None else 0
    
    @property
    def num_subhalos(self) -> int:
        return self.props.num_subhalos if self.props is not None else 0

    
@define(slots=True)
class SubBoxDataValidator: 
    all_have_props: bool
    all_have_group_ids: bool
    all_have_subhalo_ids: bool
    all_have_group_idxs: bool
    all_have_subhalo_idxs: bool

    @classmethod
    def from_sub_box_data(cls, data: dict[int, SubBox]) -> SubBoxDataValidator:
        if all_have_props := all(
            sub_box.props is not None for sub_box in data.values()
        ):
            
            return cls(
                all_have_props=all_have_props,
                all_have_group_ids = all(
                    sub_box.props.group_ids.size > 0 for sub_box in data.values()
                ),
                all_have_subhalo_ids = all(
                    sub_box.props.subhalo_ids.size > 0 for sub_box in data.values()
                ),
                all_have_group_idxs = all(
                    sub_box.props.group_indices.size > 0 for sub_box in data.values()
                ),
                all_have_subhalo_idxs = all(
                    sub_box.props.subhalo_indices.size > 0 for sub_box in data.values()
                )
            )
        else:
            return cls(
                all_have_props=all_have_props,
                all_have_group_ids=False,
                all_have_subhalo_ids=False,
                all_have_group_idxs=False,
                all_have_subhalo_idxs=False
            )

    @property
    def all_contain_groups(self) -> bool:
        return self.all_have_group_ids and self.all_have_group_idxs
    
    @property
    def all_contain_subhalos(self) -> bool:
        return self.all_have_subhalo_ids and self.all_have_subhalo_idxs

@define(slots=True)
class SubBoxes:
    global_box_size: float
    data: dict[int, SubBox]
    validator: SubBoxDataValidator = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self.validator = SubBoxDataValidator.from_sub_box_data(self.data)

    def __repr__(self) -> str:
        return (
            f"SubBoxes(global_box_size={self.global_box_size} cMpc/h, "
            f"num_sub_boxes={self.num_sub_boxes})"
        )
    @classmethod
    def empty_initialize(cls, box_size: float, num_splits: int) -> SubBoxes:
        """Initialize empty sub boxes."""
        sub_boxes = create_sub_boxes(box_size, num_splits)
        return cls(global_box_size=box_size, data=sub_boxes)
    
    @classmethod
    def from_box_particles(
            cls,
            box_size: float,
            num_splits: int,
            particle_ids: np.ndarray,
            particle_coordinates: np.ndarray
        ) -> SubBoxes:

        sub_boxes = create_sub_boxes(box_size, num_splits)
        add_sub_box_particles(particle_ids, particle_coordinates, sub_boxes)

        return cls(global_box_size=box_size, data=sub_boxes)

    @classmethod
    def from_fof_groups(
            cls,
            box_size: float,
            num_splits: int,
            group_ids: np.ndarray, 
            group_positions: np.ndarray,
            subhalo_ids: np.ndarray = np.empty(0),
            subhalo_positions: np.ndarray = np.empty(0),
        ) -> SubBoxes:

        sub_boxes = create_sub_boxes(box_size, num_splits)

        fill_sub_boxes(
            sub_boxes=sub_boxes, 
            group_ids=group_ids, 
            group_positions=group_positions, 
            subhalo_ids=subhalo_ids, 
            subhalo_positions=subhalo_positions
        )

        return cls(global_box_size=box_size, data=sub_boxes)
    
    @classmethod
    def from_subhalos(  
            cls,
            box_size: float,
            num_splits: int,
            subhalo_ids: np.ndarray,
            subhalo_positions: np.ndarray
        ) -> SubBoxes:

        sub_boxes = create_sub_boxes(box_size, num_splits)
        fill_sub_boxes(
            sub_boxes=sub_boxes,
            subhalo_ids=subhalo_ids,
            subhalo_positions=subhalo_positions
        )

        return cls(global_box_size=box_size, data=sub_boxes)

    
    def add_particles(self, particles: ParticleCollection) -> None:
        add_sub_box_particles(self.data, particles)

    def add_subhalos(self, ids: np.ndarray, positions: np.ndarray) -> None:
        
        for sub_box in self.data.values():
            in_box_mask = sub_box.is_in_box(positions)
            sub_box.props.subhalo_ids = ids[in_box_mask]
            sub_box.props.subhalo_indices = np.where(in_box_mask)[0]

    def add_groups(self, ids: np.ndarray, positions: np.ndarray) -> None:
            
        for sub_box in self.data.values():
            in_box_mask = sub_box.is_in_box(positions)
            sub_box.props.group_ids = ids[in_box_mask]
            sub_box.props.group_indices = np.where(in_box_mask)[0]

    @property
    def num_sub_boxes(self) -> int:
        return len(self.data)
    
    @property
    def group_ids(self) -> list[np.ndarray]:
        pdb.set_trace()
        if self.validator.all_contain_groups:
            return [sub_box.props.group_ids for sub_box in self.data.values()]
        else:
            return []
    
    @property
    def subhalo_ids(self) -> list[np.ndarray]:
        if self.validator.all_contain_subhalos:
            return [sub_box.props.subhalo_ids for sub_box in self.data.values()]
        else:
            return []
    
    @property
    def group_indices(self) -> list[np.ndarray]:
        if self.validator.all_contain_groups:
            return [sub_box.props.group_indices for sub_box in self.data.values()]
        else:
            return []
    
    @property
    def subhalo_indices(self) -> list[np.ndarray]:
        if self.validator.all_contain_subhalos:
            return [sub_box.props.subhalo_indices for sub_box in self.data.values()]
        else:
            return []
        
    @property
    def as_dict(self) -> dict[int, list[tuple[float, float]]]:
        return {
            sub_box_id : [
                (sub_box.x_range[0], sub_box.x_range[1]),
                (sub_box.y_range[0], sub_box.y_range[1]),
                (sub_box.z_range[0], sub_box.z_range[1])
            ]
            for sub_box_id, sub_box in self.data.items()
        }

    def get_comoving_as_dict(self, scale_factor: float) -> dict[int, list[tuple[float, float]]]:

        def make_comoving_tuple(value: tuple[float, float]) -> tuple[float, float]:
            return (value[0] / scale_factor, value[1] / scale_factor)

        return {
            sub_box_id : [
                make_comoving_tuple(sub_box.x_range),
                make_comoving_tuple(sub_box.y_range),
                make_comoving_tuple(sub_box.z_range)
            ]
            for sub_box_id, sub_box in self.data.items()
        }
    
    def get_jackknife_subbox_samples(
            self,
            coordinates: np.ndarray,
            return_as_mask: bool = True
        ) -> list[np.ndarray]:

        return get_jackknife_subbox_samples(
            sub_box_info=self.as_dict,
            coordinates=coordinates,
            return_as_mask=return_as_mask
        )
    
    def get_sub_box_samples_id_array(self, coordinates: np.ndarray) -> np.ndarray:

        return get_subbox_samples_id_array(
            sub_box_info=self.as_dict,
            coordinates=coordinates
        )
        
    
    def print_sub_box_info(self) -> None:

        num_groups, num_particles, total_volume = 0, 0, 0

        for i, sub_box in self.data.items():

            print(f"Sub Box {i}")
            print(f"{sub_box.num_groups = }")
            print(f"{sub_box.num_subhalos = }")
            print(f"{sub_box.size = }")
            print(f"{sub_box.volume = }\n")

            num_groups += sub_box.num_groups
            num_particles += sub_box.num_particles
            total_volume += sub_box.volume


        print(f"\nThere are {self.num_sub_boxes} sub boxes")
        print(f"{num_groups = }")
        print(f"{num_particles = }")
        print(f"{total_volume = }")

    
    
def get_sub_box_bounds(
        box_size: float, num_splits: int, sub_box_index: int
    ) -> dict[str, tuple[float, float]]:

    # Calculate the size of each sub box
    sub_box_size = box_size / (num_splits + 1)

    # Find the sub-box's position in each dimension
    x_pos = sub_box_index % (num_splits + 1)
    y_pos = (sub_box_index // (num_splits + 1)) % (num_splits + 1)
    z_pos = sub_box_index // ((num_splits + 1) ** 2)

    # Calculate and return bounds for each dimension
    return {
        "x": (x_pos * sub_box_size, (x_pos + 1) * sub_box_size),
        "y": (y_pos * sub_box_size, (y_pos + 1) * sub_box_size),
        "z": (z_pos * sub_box_size, (z_pos + 1) * sub_box_size)
    }

def create_sub_boxes(box_size: float, num_splits: int) -> dict[int, SubBox]:

    assert num_splits >= 0, "Number of splits must be non-negative"

    sub_boxes = {}
    num_sub_boxes = (num_splits + 1) ** 3
    for i in range(num_sub_boxes):
        bounds = get_sub_box_bounds(box_size, num_splits, i)
        sub_boxes[i] = SubBox(
            size = (box_size / (num_splits + 1)), 
            x_range=bounds["x"],
            y_range=bounds["y"],
            z_range=bounds["z"]
        )
    return sub_boxes

def fill_sub_boxes(
        sub_boxes: dict[int, SubBox],
        group_ids: np.ndarray = np.empty(0), 
        group_positions: np.ndarray = np.empty(0),
        subhalo_ids: np.ndarray = np.empty(0),
        subhalo_positions: np.ndarray = np.empty(0),
    ) -> None:

    for sub_box in sub_boxes.values():

        if group_ids.size == 0 and subhalo_ids.size == 0:
            print("No data to fill sub boxes with")
            break

        if group_ids.size > 0:
            groups_in_box = sub_box.is_in_box(group_positions)
            sub_box.props.group_ids = group_ids[groups_in_box]
            sub_box.props.group_indices = np.where(groups_in_box)[0]

        if subhalo_ids.size > 0:
            subhalos_in_box = sub_box.is_in_box(subhalo_positions)
            sub_box.props.subhalo_ids = subhalo_ids[subhalos_in_box]
            sub_box.props.subhalo_indices = np.where(subhalos_in_box)[0]

def add_sub_box_particles(
        sub_boxes: dict[int, SubBox], particles: ParticleCollection
    ) -> None:

    start = time()

    for box in sub_boxes.values():
        in_box_mask = box.is_in_box(particles.coordinates)
        box.add_particle_data(particles.ids[in_box_mask], np.where(in_box_mask)[0])

    print("Particles added in", time() - start, "seconds")

def add_sub_box_particles(
        particle_ids: np.ndarray,
        particle_coordinates: np.ndarray,
        sub_boxes: dict[int, SubBox]
    ) -> None:

    start = time()

    for box in sub_boxes.values():
        in_box_mask = box.is_in_box(particle_coordinates)
        box.add_particle_data(particles_ids[in_box_mask], np.where(in_box_mask)[0])

    print("Particles added in", time() - start, "seconds")

def get_subbox_samples_id_array(
        sub_box_info: dict[int, list[tuple[float, float]]],  # {1: [(x_i, x_f), (y_i, y_f), (z_i, z_f)], ...}
        coordinates: np.ndarray,
    ) -> np.ndarray:
    """
    Assign each point to a sub-box ID with robust boundary handling.

    Rule:
      - Use half-open intervals [lo, hi) for all boxes,
      - but make the interval closed on the *global* maximum edge per axis, i.e. [lo, hi_max].
      - Add a tiny epsilon so floating-point jitter doesn’t cause gaps.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"`coordinates` must have shape (N, 3); got {coordinates.shape}")

    N = coordinates.shape[0]
    ids = np.full(N, -1, dtype=int)

    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Find global mins/maxes over the sub-box definitions
    x_max = max(b[0][1] for b in sub_box_info.values())
    y_max = max(b[1][1] for b in sub_box_info.values())
    z_max = max(b[2][1] for b in sub_box_info.values())

    # A small tolerance relative to the full extent to absorb float noise
    # (scale by range so it works for tiny/huge boxes alike)
    span = max(x_max, y_max, z_max)  # assumes coords start near 0; fine for [0, L]
    eps = 1e-12 * max(1.0, span)

    # To keep behavior stable regardless of dict ordering, iterate in key order
    for subbox_id, bounds in sorted(sub_box_info.items(), key=lambda x: x[0]):
        (x0, x1), (y0, y1), (z0, z1) = bounds

        # Half-open by default; closed on the global max edge for that axis
        x_hi_inclusive = np.isclose(x1, x_max, rtol=0.0, atol=eps)
        y_hi_inclusive = np.isclose(y1, y_max, rtol=0.0, atol=eps)
        z_hi_inclusive = np.isclose(z1, z_max, rtol=0.0, atol=eps)

        xm = (x >= x0 - eps) & ((x < x1 - eps) | (x_hi_inclusive & (x <= x1 + eps)))
        ym = (y >= y0 - eps) & ((y < y1 - eps) | (y_hi_inclusive & (y <= y1 + eps)))
        zm = (z >= z0 - eps) & ((z < z1 - eps) | (z_hi_inclusive & (z <= z1 + eps)))

        mask = xm & ym & zm
        # First-match-wins avoids double-assignments on shared faces
        ids[(ids == -1) & mask] = subbox_id

    return ids