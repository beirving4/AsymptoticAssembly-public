import pdb
from functools import partial
from collections import OrderedDict

from ..cosmo.config import CosmoType 

TOY_MODEL_A_COSMO_MASSES = OrderedDict({
    256: OrderedDict({1024: 5.46959E12}),
    512: OrderedDict({
        4: 4.07516E4,
        16: 2.6081E6,
        32: 2.08648E7,
        128: 1.33535E9,
        256: 1.06828E10,
        512: 8.54623E10,
        1024: 6.83699E11,
    }),
    1024 : OrderedDict(
        {
            128: 1.66919E8,
            1024: 8.54623E10,
            2048: 6.83699E11,
        }
    ),
    2048 : OrderedDict({128: 2.08648E7})
})


TOY_MODEL_B_COSMO_MASSES = OrderedDict({
    256: OrderedDict({256: 7.65954E10}),
    512: OrderedDict({512: 7.65954E10}), 
    1024: OrderedDict({1024: 7.65954E10}), 
})

PRIMARY_COSMO_MASSES = OrderedDict({
    256: OrderedDict({256: 8.49073829E10}),
    1024: OrderedDict({
        32: 2.591170e+06, 
        128: 1.658350e+08,
        512: 1.061340e+10,
        2048: 6.792590e+11,
    }),
})

# Planck2018
PLANCK18_COSMO_MASSES = OrderedDict({
    512: OrderedDict({
        4: 4.11618E4,
        32: 2.10748E7,
        128: 1.34879E9,
        256: 1.07903E10,
        512: 8.63225E10,
        1024: 6.9058E11,
        2048: 5.52464E12,
        4096: 4.41971E13,
        8192: 3.53577E14
    }),
}) 

def get_particle_mass_set(cosmo_name: str) -> OrderedDict:
    match cosmo_name.lower():
        case CosmoType.PRIMARY.value:
            return PRIMARY_COSMO_MASSES
        case CosmoType.TOY_MODEL_A.value:
            return TOY_MODEL_A_COSMO_MASSES
        case CosmoType.TOY_MODEL_B.value:
            return TOY_MODEL_B_COSMO_MASSES
        case CosmoType.PLANCK.value:
            return PLANCK18_COSMO_MASSES
        case _:
            raise ValueError(f"{cosmo_name} is not a valid cosmology name")


def get_particle_mass(
        cosmo_name: str, box_size: int, num_particles: int
    ) -> float:
    return get_particle_mass_set(cosmo_name)[num_particles][box_size]
    
def get_min_resolved_mass(
        cosmo_name: str, 
        box_size: int, 
        min_num_resolved_particles: int,
        num_particles_in_sim: int, 
    ) -> float:

    particle_mass = get_particle_mass(
        cosmo_name=cosmo_name, 
        box_size=box_size, 
        num_particles=num_particles_in_sim
    )

    return min_num_resolved_particles * particle_mass


@partial
def get_toy_model_a_particle_mass(num_particles: int, box_size: int) -> float:
    return get_particle_mass("toy_model_a", num_particles, box_size)

@partial
def get_toy_model_a_min_resolved_mass(
        box_size: int, 
        min_num_resolved_particles: int, 
        num_particles_in_sim: int
    ) -> float:
    return get_min_resolved_mass(
        "toy_model_a", 
        box_size, 
        min_num_resolved_particles, 
        num_particles_in_sim
    )

@partial
def get_toy_model_b_particle_mass(box_size: int) -> float:
    return get_particle_mass("toy_model_b", box_size, num_particles=-1)

@partial
def get_toy_model_b_min_resolved_mass(
        box_size: int, 
        min_num_resolved_particles: int, 
        num_particles_in_sim: int
    ) -> float:
    return get_min_resolved_mass(
        "toy_model_b", 
        box_size, 
        min_num_resolved_particles, 
        num_particles_in_sim
    )

@partial
def get_primary_particle_mass(num_particles: int, box_size: int) -> float:
    return get_particle_mass("primary", num_particles, box_size)

@partial
def get_primary_min_resolved_mass(
        box_size: int, 
        min_num_resolved_particles: int, 
        num_particles_in_sim: int
    ) -> float:
    return get_min_resolved_mass(
        "primary", 
        box_size, 
        min_num_resolved_particles, 
        num_particles_in_sim
    )

@partial
def get_planck18_particle_mass(num_particles: int, box_size: int) -> float:
    return get_particle_mass("planck", num_particles, box_size)

@partial
def get_planck18_min_resolved_mass(
        box_size: int, 
        min_num_resolved_particles: int, 
        num_particles_in_sim: int
    ) -> float:
    return get_min_resolved_mass(
        "planck", 
        box_size, 
        min_num_resolved_particles, 
        num_particles_in_sim
    )

