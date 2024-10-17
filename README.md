# tss

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/trinhhe/tss/actions/workflows/ci.yaml/badge.svg)](https://github.com/trinhhe/tss/actions/workflows/ci.yaml)

> long-range conditional time series generation with autoregressive models

## About

TODO

## Example usage

TODO

## Installation

To install the latest GitHub <RELEASE>, just call the following on the
command line:

```bash
pip install git+https://github.com/trinhhe/tss@<RELEASE>
```

## Development

First install [`rye`](https://rye.astral.sh/) and [`uv`](https://pypi.org/project/uv/) then call
```shell
rye sync
source .venv/bin/activate
uv pip install mamba-ssm --no-build-isolation
```

You can then add/remove dependencies using
```shell
rye add/remove <dependency>
```

This will create lock files that can be used to reproduce your experimental results..


## Author

Henry Trinh <a href="mailto:trinhhe@ethz.ch">trinhhe@ethz.ch</a>
