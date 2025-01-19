# pymdp-cookbook
A collection of examples demonstrating active inference using [pymdp](https://github.com/infer-actively/pymdp),
specifically the `v1.0.0_alpha` branch.

These will largely focus on how to set up a generative model to represent real world problems

## Setup
you can use pip and venv as below
```bash
python3 -m venv venv
source .venv/bin/activate
pip install -r requirements.txt
```
or you can use `uv` with the `pyproject.toml` file.

## Execution
You can run any of the examples independently with e.g. `python3 examples/perception.py`
