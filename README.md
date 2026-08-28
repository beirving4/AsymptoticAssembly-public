# AsymptoticAssembly

AsymptoticAssembly is a Python package for studying the assembly histories,
mass evolution, density fields, abundance, and merger-tree structure of dark
matter halos in cosmological N-body simulations.

This repository is the public thesis-release snapshot. It contains the
portable scientific package and intentionally omits the author's private
research drivers, cluster layouts, scheduler templates, and infrastructure
configuration. Development and validation are continuing after the thesis
deposit; the current release is therefore marked as a pre-release.

## Status

- Version: `0.1.0.dev0`
- Python: 3.12 or newer
- License: MIT
- Release stage: thesis pre-release
- Full multi-environment and large-scale cluster validation: in progress

The API may change before the first stable release. Numerical results used in
research should record the exact release tag and configuration.

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/beirving4/AsymptoticAssembly-public.git
cd AsymptoticAssembly-public
uv sync
```

To install the composed workflow interface as well:

```bash
uv sync --extra workflows
```

Optional dependency groups are also available for clustering, field
transforms, accelerated profile work, and visualization. See
`pyproject.toml` for their exact contents.

## Public scope

The public snapshot includes:

- halo assembly and mass-history analysis;
- abundance and mass-function models;
- density, correlation, power-spectrum, and particle-property tools;
- density-PDF and thesis-model fitting boundaries;
- merger-tree interfaces, including the Tessera-oriented backend boundary;
- portable Hydra/OmegaConf workflow configuration as an optional extra.

Research-only post-processing scripts and machine-specific data discovery are
not part of this repository. Users supply their own data locations and may
build project-specific workflow adapters around the public package.

Tessera is a separately developed compiled backend. Functionality requiring
it reports a clear error when the backend is unavailable; it is not silently
replaced by a slower pure-Python implementation.

## Reproducibility and validation

The package favors NumPy/SciPy and compiled backends for high-throughput
scientific workloads. The public-release cleanup preserves scientific
regression contracts while improving packaging boundaries, typing, and
portability. This snapshot has passed the focused scientific regression suites
used during cleanup. Remaining release gates—clean-environment installation,
the complete suite across supported environments, installed-wheel checks, and
large-scale performance validation—are tracked as post-deposit work and are
not claimed complete here.

## Citation

Zenodo citation metadata and the DOI will be attached to the GitHub release.
Until then, cite the repository and the exact version tag. A machine-readable
`CITATION.cff` is included.

## Contributing

Issues and focused pull requests are welcome. Please include a minimal
reproducer, the Python version, dependency versions, and—when relevant—the
simulation format and scale. Scientific changes should include regression
tests or a documented numerical comparison.

## Author

Bryen Irving
