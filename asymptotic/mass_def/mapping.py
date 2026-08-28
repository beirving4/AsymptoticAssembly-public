from __future__ import annotations

from attrs import define, fields

@define(slots=True)
class DensityThresholdProperty:
    name: str 
    eqn: str

@define
class DensityThresholdMap:
    name: str
    key: str
    title: str
    display: str
    mass: DensityThresholdProperty
    peak_height: DensityThresholdProperty
    radius: DensityThresholdProperty

fof_map = DensityThresholdMap(
    name="fof",
    key="fof",
    title="Friends-of-Friends",
    display=r"$\Delta_{\rm FoF}$",
    mass=DensityThresholdProperty("masses", r"$M_{\rm FoF}$"),
    peak_height=DensityThresholdProperty("peak_height", r"$\nu_{\rm FoF}$"),
    radius=DensityThresholdProperty("radii", r"$R_{\rm FoF}$")
)

subfind_map = DensityThresholdMap(
    name="subfind",
    key="subfind",
    title="Subfind",
    display=r"$\Delta_{\rm subhalo}$",
    mass=DensityThresholdProperty("SubhaloMass", r"$M_{\rm subhalo}$"),
    peak_height=DensityThresholdProperty("peakSubhalo", r"$\nu_{\rm subhalo}$"),
    radius=DensityThresholdProperty("SubhaloRadius", r"$R_{\rm subhalo}$")
)

crit200_map = DensityThresholdMap(
    name="crit200",
    key="200c",
    title="Critical 200",
    display=r"$\Delta_{\rm 200c}$",
    mass=DensityThresholdProperty("masses200c", r"$M_{\rm 200c}$"),
    peak_height=DensityThresholdProperty("peak200c", r"$\nu_{\rm 200c}$"),
    radius=DensityThresholdProperty("radii200c", r"$R_{\rm 200c}$")
)

crit500_map = DensityThresholdMap(
    name="crit500",
    key="500c",
    title="Critical 500",
    display=r"$\Delta_{\rm 500c}$",
    mass=DensityThresholdProperty("masses500c", r"$M_{\rm 500c}$"),
    peak_height=DensityThresholdProperty("peak500c", r"$\nu_{\rm 500c}$"),
    radius=DensityThresholdProperty("radii500c", r"$R_{\rm 500c}$")
)

mean200_map = DensityThresholdMap(
    name="mean200",
    key="200m",
    title="Mean 200",
    display=r"$\Delta_{\rm 200m}$",
    mass=DensityThresholdProperty("masses200m", r"$M_{\rm 200m}$"),
    peak_height=DensityThresholdProperty("peak200m", r"$\nu_{\rm 200m}$"),
    radius=DensityThresholdProperty("radii200m", r"$R_{\rm 200m}$")
)

virial_map = DensityThresholdMap(
    name="virial",
    key="vir",
    title="Virial",
    display=r"$\Delta_{\rm vir}$",
    mass=DensityThresholdProperty("massesVirial", r"$M_{\rm vir}$"),
    peak_height=DensityThresholdProperty("peakVirial", r"$\nu_{\rm vir}$"),
    radius=DensityThresholdProperty("radiiVirial", r"$R_{\rm vir}$")
)

splashback_map = DensityThresholdMap(
    name="splashback",
    key="sp",
    title="Splashback",
    display=r"$\Delta_{\rm sp}$",
    mass=DensityThresholdProperty("massesSplashback", r"$M_{\rm sp}$"),
    peak_height=DensityThresholdProperty("peakSplashback", r"$\nu_{\rm sp}$"),
    radius=DensityThresholdProperty("radiiSplashback", r"$R_{\rm sp}$")
)

scale_map = DensityThresholdMap(
    name="scale",
    key="scale",
    title="Scale",
    display=r"$\Delta_{\rm s}$",
    mass=DensityThresholdProperty("massesScale", r"$M_{\rm s}$"),
    peak_height=DensityThresholdProperty("peakScale", r"$\nu_{\rm s}$"),
    radius=DensityThresholdProperty("radiiScale", r"$R_{\rm s}$")
)

four_scale_map = DensityThresholdMap(
    name="four_scale",
    key="<4s",
    title="Four Scale",
    display=r"$\Delta_{\rm s}$",
    mass=DensityThresholdProperty("massesFourScale", r"$M_{\rm <4s}$"),
    peak_height=DensityThresholdProperty("peakFourScale", r"$\nu_{\rm <4s}$"),
    radius=DensityThresholdProperty("radiiFourScale", r"$R_{\rm <4s}$")
)

max_circ_map = DensityThresholdMap(
    name="max_circ",
    key="max",
    title="MaxCircular",
    display=r"$\Delta_{\rm max}$",
    mass=DensityThresholdProperty("massesSplashback", r"$M_{\rm max}$"),
    peak_height=DensityThresholdProperty("peakSplashback", r"$\nu_{\rm max}$"),
    radius=DensityThresholdProperty("radiiSplashback", r"$R_{\rm max}$")
)

bound_map = DensityThresholdMap(
    name="bound",
    key="bound",
    title="Bounded",
    display=r"$\Delta_{\rm bound}$",
    mass=DensityThresholdProperty("massesSplashback", r"$M_{\rm bound}$"),
    peak_height=DensityThresholdProperty("peakSplashback", r"$\nu_{\rm bound}$"),
    radius=DensityThresholdProperty("radiiSplashback", r"$R_{\rm bound}$")
)

# Kinematic outer-boundary milestones derived from <v_r>(r). Populated
# by FOFGroups.add_solo_boundaries from solo's mean_vr-based extractor
# (compute_outer_boundary_milestones in
# AsymptoticAssembly.asymptotic.utils.kinematic_milestones).
hydrostatic_map = DensityThresholdMap(
    name="hydrostatic",
    key="hs",
    title="Hydrostatic",
    display=r"$\Delta_{\rm hs}$",
    mass=DensityThresholdProperty("massesHydrostatic", r"$M_{\rm hs}$"),
    peak_height=DensityThresholdProperty("peakHydrostatic", r"$\nu_{\rm hs}$"),
    radius=DensityThresholdProperty("radiiHydrostatic", r"$R_{\rm hs}$")
)

turnaround_map = DensityThresholdMap(
    name="turnaround",
    key="ta",
    title="Turnaround",
    display=r"$\Delta_{\rm ta}$",
    mass=DensityThresholdProperty("massesTurnaround", r"$M_{\rm ta}$"),
    peak_height=DensityThresholdProperty("peakTurnaround", r"$\nu_{\rm ta}$"),
    radius=DensityThresholdProperty("radiiTurnaround", r"$R_{\rm ta}$")
)

infall_map = DensityThresholdMap(
    name="infall",
    key="inf",
    title="Infall",
    display=r"$\Delta_{\rm inf}$",
    mass=DensityThresholdProperty("massesInfall", r"$M_{\rm inf}$"),
    peak_height=DensityThresholdProperty("peakInfall", r"$\nu_{\rm inf}$"),
    radius=DensityThresholdProperty("radiiInfall", r"$R_{\rm inf}$")
)

# Marginally-bound shell. Pavlidou-Tomaras max-turnaround surface
# (<ρ_enc> = 2·ρ_Λ). Sourced from the SO-bisection lambda_so when the
# solo run included --so_catalog (authoritative); falls back per-halo to
# the profile-interp lambda otherwise.
marginally_bound_map = DensityThresholdMap(
    name="marginally_bound",
    key="marg_bound",
    title="Marginally Bound",
    display=r"$\Delta_{\Lambda}$",
    mass=DensityThresholdProperty("massesMargBound", r"$M_{\Lambda}$"),
    peak_height=DensityThresholdProperty("peakMargBound", r"$\nu_{\Lambda}$"),
    radius=DensityThresholdProperty("radiiMargBound", r"$R_{\Lambda}$")
)

asymptotic_map = DensityThresholdMap(
    name="asymptotic",
    key="asymptotic",
    title="Asymptotic",
    display=r"$\Delta_{\infty}$",
    mass=DensityThresholdProperty("massesAsymptotic", r"$M_{\infty}$"),
    peak_height=DensityThresholdProperty("peakAsymptotic", r"$\nu_{\infty}$"),
    radius=DensityThresholdProperty("radiiAsymptotic", r"$R_{\infty}$")
)

@define(slots=True)
class MassDefMapping: 
    fof: DensityThresholdMap = fof_map
    subfind: DensityThresholdMap = subfind_map

    crit200: DensityThresholdMap = crit200_map
    crit500: DensityThresholdMap = crit500_map
    mean200: DensityThresholdMap = mean200_map
    virial: DensityThresholdMap = virial_map

    scale: DensityThresholdMap = scale_map
    four_scale: DensityThresholdMap = four_scale_map
    max_circ: DensityThresholdMap = max_circ_map
    bound: DensityThresholdMap = bound_map
    splashback: DensityThresholdMap = splashback_map

    # Kinematic outer-boundary milestones (mean_vr-derived).
    hydrostatic: DensityThresholdMap = hydrostatic_map
    turnaround: DensityThresholdMap = turnaround_map
    infall: DensityThresholdMap = infall_map
    marginally_bound: DensityThresholdMap = marginally_bound_map

    asymptotic: DensityThresholdMap = asymptotic_map


    def __getitem__(self, attr_name: str) -> DensityThresholdMap:
        return getattr(self, attr_name)
    
    def use_key(self, key: str) -> DensityThresholdMap | None:
        for attr_name in self.__slots__:
            if getattr(self, attr_name).key == key:
                return self[attr_name]

    @property
    def fields(self) -> list[str]:
        return [attr.name for attr in fields(self.__class__)]
    

global_mass_def_mapping = MassDefMapping()
