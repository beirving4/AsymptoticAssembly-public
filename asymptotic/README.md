# asymptotic — Core Python Package

The `asymptotic` package contains all library code for analyzing dark matter halo assembly in cosmological N-body simulations. Each submodule handles a distinct aspect of the analysis pipeline.

## Module Reference

### `analysis/` — Pipeline Entry Point

The Hydra-based analysis pipeline that orchestrates the full analysis workflow.

| File | Description |
|------|-------------|
| `main.py` | Entry point. Uses `@hydra.main` to load configuration and instantiate `SimulationData` for the analysis run. |
| `resume.py` | Resume interrupted analyses from checkpoints. Supports resuming assembly tracking and historical data analysis from a specified snapshot. |
| `config.yaml` | Hydra configuration file defining simulation name, I/O paths, cosmology, merger tree settings, profile parameters, mass function bins, and checkpoint behavior. |

**Key configuration options** (`config.yaml`):
- `analyze_assemblies` / `analyze_historical_data`: Toggle which analyses to run
- `merger_tree.top_k`: Number of most massive halos to track (default: 2000)
- `snapshot.profiles.num_radial_bins`: Radial bins for density profiles (default: 50)
- `snapshot.mass_function.num_mass_bins`: Bins for mass function estimation (default: 100)
- `checkpoint.resume`: Resume from a previous run

---

### `assembly/` — Halo Assembly Histories

Tracks how halos grow through accretion and mergers over cosmic time.

| File | Description |
|------|-------------|
| `histories.py` | `AssemblyHistory` (single halo: accretion + pseudo-evolution + mergers), `AssemblyHistories` (collection), `MassAssemblyHistories` (per mass definition). |
| `accretion.py` | `AccretionHistory` / `AccretionHistories` — mass accretion history (MAH) tracking for individual and population-level analysis. |
| `mergers.py` | `MergerHistory` / `MergerHistories` — merger event records and rates. |
| `trees.py` | Generic saved-order tree machinery: `save_merger_tree_info` + order loaders, `GroupTree`/`GroupTreeData` order-based construction. The ytree-arbor helpers were retired to `postprocessing/legacy_ytree/` (A3b2); the shipped module no longer references ytree. |

---

### `bounded/` — FOF Groups and Subhalos

Core data structures representing gravitationally bound objects identified in simulations.

| File | Description |
|------|-------------|
| `fof.py` | `FOFGroup` (single FOF halo with mass definitions, subhalos, particles, shapes, anisotropy), `FOFGroups` (collection at one snapshot), `FOFGroupEvoData` (evolution over time). Order-based history extraction (`get_main_progenitor_branch*`) works from precomputed evo paths; the ytree-arbor linkage was retired to `postprocessing/legacy_ytree/` (A3b2). |
| `subhalo.py` | `Subhalos`, `SubhaloMap`, `SubhaloMapping`, `SubhaloMappingEvo` — subhalo tracking and group-to-subhalo mappings across snapshots. |
| `objects.py` | `BoundedObject` base class, `BoundedObjectEvolutionData` — generic bounded structure with time evolution. |
| `population.py` | `HaloPopulation`, `HaloPopulationEvolutionData` — population-level statistics (mean mass, counts, distributions). |
| `pseudo.py` | Pseudo-evolution modeling — mass changes from evolving reference density without physical accretion. |
| `bias.py` | Halo bias calculations relative to the background matter distribution. |
| `tracker.py` | Cross-snapshot halo/subhalo tracking. |
| `ode.py` | ODE solvers for halo evolution models. |

---

### `cosmo/` — Cosmological Framework

Cosmological parameter definitions and model calculations, wrapping the Colossus library.

| File | Description |
|------|-------------|
| `parameters.py` | Pre-defined cosmological parameter sets: `PRIMARY_COSMO_PARAMETERS` (H0=68.0, Omega_m=0.306, sigma_8=0.804), `PLANCK18_COSMO_PARAMETERS`, `TOY_MODEL_A_COSMO_PARAMETERS`, `TOY_MODEL_B_COSMO_PARAMETERS`. Each includes H0, Omega_m, Omega_b, Omega_L, sigma_8, n_s, w0, wa, transfer function specification. |
| `model.py` | `Cosmology` class — wraps Colossus for sigma(M,z), power spectrum P(k), growth factor D(z), and related calculations. Handles edge cases at high/low redshifts. |
| `config.py` | `CosmoType`, `TransferFn` enums — configuration for cosmology model selection. |
| `transfer.py` | Transfer function implementations: Eisenstein98, CAMB, CLASS, Efstathiou. |

---

### `mass_def/` — Mass Definition Framework

Supports simultaneous analysis under 10+ mass definitions.

| File | Description |
|------|-------------|
| `base.py` | `MassDefinitionDependentType` — abstract base class for any quantity that varies with mass definition. Defines standardized colors, line styles, and marker styles for each definition (fof, subfind, crit200, crit500, mean200, virial, splashback, asymptotic, etc.). |
| `data.py` | `MassDefinition` (mass, radius, density, circular velocity, Hubble parameter at a single definition), `MassDefinitions` (collection of all definitions for one object). |
| `mapping.py` | Mapping logic between definition names and their physical properties. |

**Supported definitions**: fof, subfind, crit200, crit500, mean200, virial, scale, four_scale, max_circ, bound, splashback, asymptotic.

---

### `particles/` — Particle-Level Analysis

Detailed analysis of individual dark matter particles and their collective properties.

| File | Description |
|------|-------------|
| `identify.py` | `ParticleIdentifier` — particle ID management and lookup operations for tracking particles across snapshots. |
| `collection.py` | `ParticleCollection`, `BoxParticleCollection`, `BoundedParticleCollection` — manages arrays of positions, velocities, masses. Includes KDTree construction and 2PCF computation. |
| `properties.py` | `ParticleCollectionProperties`, `BoxParticleProperties`, `BoundedParticleProperties` — derived physical properties of particle ensembles. |
| `nbody.py` | `NBodyParticle` — combines particle ID with physical properties. |
| `spin.py` | Angular momentum and spin parameter (lambda_spin) calculations. |
| `shapes.py` | `ShapeParameterData`, `ShapeProfileData`, `PopulationShapeParameterData` — axis ratios (c/a, b/a) from the inertia tensor. |
| `anisotropy.py` | `AnisotropyParameterData`, `AnisotropyProfileData`, `PopulationAnisotropyParameterData` — velocity anisotropy beta(r) profiles. |
| `orbital.py` | Orbital mechanics — periapsis, apoapsis, eccentricity calculations for orbiting substructure. |
| `composition.py` | Particle composition analysis: in-situ vs. accreted material classification. |
| `tracking.py` | Particle trajectory tracking across simulation snapshots. |
| `masses.py` | Minimum resolved mass computation from particle data. |
| `composite.py` | Composite particle metrics combining multiple properties. |
| `bounded.py` | Bounded region particle evolution and associated statistics. |
| `viz.py` | Particle visualization utilities. |

---

### `distribution/` — Radial Distributions and Profiles

Radial binning, density profiles, and phase-space analysis.

| File | Description |
|------|-------------|
| `data.py` | `DistributionData` — base class for radially binned data. |
| `mass.py` | `MassDistributionData` — enclosed mass as a function of radius. |
| `profiles.py` | Radial density and mass profiles with configurable binning. |
| `slope.py` | Density slope d(ln rho)/d(ln r) calculations; spherical overdensity (SO) radius determination for various overdensity thresholds. |
| `phase_space.py` | Phase-space (r, v_r) diagrams for halo particles. |
| `analysis.py` | Profile fitting and statistical analysis tools. |
| `evo.py` | Time evolution of distribution functions across snapshots. |
| `helpers.py` | Helper functions for binning and interpolation. |
| `units.py` | Unit conversion utilities for profile quantities. |
| `viz.py` | Distribution and profile visualization. |

---

### `fields/` — Large-Scale Structure Fields

Density field computation, power spectra, and spatial correlation functions.

| File | Description |
|------|-------------|
| `density.py` | Grid-based density field construction (using pmesh/enzo). |
| `peaks.py` | `DensityPeaks` — peak height nu(M,z) = delta_c(z)/sigma(M,z) using Colossus. Methods: `heights()`, `mass_variance()`, `jacobian()`, `multiplicity()`. |
| `power.py` | Matter power spectrum P(k) computation. |
| `two_point.py` | Two-point correlation function xi(r) via TreeCorr. |
| `correlation.py` | Correlation function data structures and containers. |
| `knn.py` | k-nearest neighbor distance distributions. |
| `filter.py` | Density field smoothing (Gaussian, top-hat filters). |
| `variance.py` | Power spectrum variance sigma^2(R) integration. |
| `stitching.py` | Combining correlation functions measured across different scale ranges. |
| `treecorr_stitching.py` | TreeCorr-specific stitching logic. |
| `accumulation.py` | Cumulative field statistics. |
| `correlation_freeze.py` | Snapshot-frozen correlation function data. |
| `viz.py` | Field visualization tools. |

---

### `abundance/` — Abundance Functions

Halo mass functions. Void size and filament length relations are out of
scope for the publication release (see `docs/packaging.md`).

| File | Description |
|------|-------------|
| `data.py` | `Abundances` (histogram + differential + cumulative), `AbundanceValue`, `AbundanceRelation`, `AbundanceData`. Supports `HALO_MASS_FUNCTION`. |
| `compute.py` | Abundance computation from particle/halo samples. |
| `mass_function.py` | Mass function binning and dn/dM estimation. |
| `halo.py` | Halo-specific abundance calculations. |
| `accumulation.py` | Cumulative abundance n(>M). |
| `evo.py` | Redshift evolution of abundance relations. |
| `triangle.py` | `CoverageTriangle` — data quality assessment for histogram coverage. |
| `type_defs.py` | `AbundanceType`, `RelationType` enums. |
| `viz.py` | Abundance function plots. |

---

### `model/` — Analytical Models and Fitting

Theoretical models and curve fitting for halo properties.

| File | Description |
|------|-------------|
| `mass_function.py` | `AsymptoticAbundanceFit` — multiplicity function f(nu) = A[(nu/b)^a + 1]exp(-c*nu^2) following Diemer (2020). Default parameters: A=0.124399, a=1.191457, b=0.337871, c=0.431710. Computes multiplicity, number density, differential dn/dlnM, cumulative n(>M). |
| `curve/mah.py` | Mass accretion history model fitting. |
| `curve/abundance.py` | Abundance curve fitting utilities. |
| `curve/distribution.py` | Distribution function model fitting. |
| `curve/fit.py` | General curve fitting infrastructure (scipy.optimize.curve_fit wrapper). |
| `evo.py` | Time evolution of model predictions. |
| `viz.py` | Model comparison and residual visualization. |

---

### `simulation/` — Simulation Data Structures

Data structures for managing simulation snapshots and time evolution.

| File | Description |
|------|-------------|
| `box.py` | Simulation box representation (periodic boundaries, box length). |
| `snapshot.py` | Snapshot metadata management (snapshot IDs, file paths). |
| `moments.py` | `Moment` (single time step: snapshot_id, scale factor a, redshift z), `MomentsInTime` (ordered collection of moments). |
| `evo.py` | `EvolutionData` — `OrderedDict` keyed by snapshot_id for storing time-evolving quantities. Base class for all evolution data. |
| `sub_box.py` | `SubBoxes` — sub-volume partitioning for jackknife resampling and convergence testing. |
| `data.py` | Simulation data containers. |
| `viz.py` | Simulation state visualization. |

---

### `studies/` — Systematic Studies

Infrastructure for running convergence tests, variance estimates, and parameter studies.

| File | Description |
|------|-------------|
| `dataloader.py` | Multi-simulation catalog loading: `get_global_mass_bin_evo()`, `get_global_size_bin_evo()`, `get_global_length_bin_evo()`. Supports `CatalogType`: HALOS, VOIDS, FILAMENTS. |
| `config.py` | Study configuration management. |
| `initialize.py` | Study initialization and setup. |
| `convergence.py` | Resolution and box-size convergence testing. |
| `variance.py` | Cosmic variance estimation across realizations. |
| `io.py` | Study I/O utilities for loading/saving results. |
| `mass_function/` | Mass function studies: `box_size.py` (box size dependence), `variance.py` (variance analysis), `cosmo.py` (cosmology dependence), `data.py` (data loading), `config.py` (study config). |
| `mah/` | Mass accretion history studies: `data.py` (MAH data loading), `helpers.py` (MAH utilities), `stats.py` (MAH statistics). |

---

### `utils/` — Utility Functions

General-purpose helpers for I/O, data loading, and analysis support.

| File | Description |
|------|-------------|
| `get_data.py` | Core data loading from Gadget4 HDF5 outputs: `get_fof_data()`, `get_fof_subfind_data()`, `load_physical_catalog_data()`. Handles batched and unbatched file I/O, MUSIC parameter parsing, snapshot directory discovery. |
| `analysis.py` | Analysis utilities: `AnalysisConfig`, `display_time()`, `get_box_particles()`, `get_fof_group_evo_data()`. |
| `bounded_study_io.py` | HDF5 I/O for bounded object studies: `save_bounded_hdf5()`, `save_bounded_chkpt_hdf5()`. |
| `directory_finder.py` | Path and directory location logic for finding simulation outputs. |
| `jackknife.py` | Jackknife resampling for error estimation. |
| `threads.py` | Thread pool management for parallel computations. |
| `code_profiler.py` | Performance profiling utilities. |
| `symspace.py` | Symmetric space calculations. |
| `freeze_out.py` | Freeze-out property computation. |
| `get_correct_corrfunc.py` | Correlation function method selection logic. |
| `load.py` | Load method aggregation. |

---

### `visualize/` — Visualization

Plotting utilities organized by analysis type.

| File | Description |
|------|-------------|
| `assembly.py` | Assembly history and MAH plots. |
| `distribution.py` | Radial profile and distribution function plots. |
| `particles.py` | Particle projection and phase-space plots. |
| `clustering.py` | Correlation function and clustering visualizations. |
| `plot_abundance.py` | Mass function and abundance relation plots. |
| `viz.py` | General visualization toolkit (shared styling, figure setup). |

---

### `analysis_old/` and `mass_function_old/` — Legacy Code

Deprecated implementations from earlier development. They are retained only as historical reference, are not loaded by the active package, and are excluded from the primary Graphify corpus. Current code should not add imports from either tree.
