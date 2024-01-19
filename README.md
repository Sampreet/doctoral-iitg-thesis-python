# Macroscopic Quantum Phenomena in Hybrid Optomechanical Systems: A Theoretical Exploration

[![Thesis Version](https://img.shields.io/badge/thesis-1.0-red?style=for-the-badge)](#)
[![Toolbox Version](https://img.shields.io/badge/qom-v1.0.1-red?style=for-the-badge)](https://sampreet.github.io/qom-docs/v1.0.1)
[![Last Updated](https://img.shields.io/github/last-commit/sampreet/doctoral-iitg-thesis-python?style=for-the-badge)](https://github.com/sampreet/doctoral-iitg-thesis-python/blob/master/CHANGELOG.md)

> Python files for the Ph.D. thesis submitted to IIT Guwahati.

## Structure of the Repository

```
ROOT_DIR/
|
│───scripts/
│   ├───bar/
│   │   ├───baz.py
│   │   └───...
│   └───...
|
├───systems/
│   ├───Foo.py
│   └───...
│
├───.gitignore
├───CHANGELOG.md
└───README.md
```

Here, `foo` represents the module or system and `bar` represents the version.

## Execution

### Installing Dependencies

All numerical data and plots are obtained using the [Quantum Optomechanics Toolbox](https://github.com/sampreet/qom), an open-source Python framework to simulate optomechanical systems.
Refer to the [QOM toolbox documentation](https://sampreet.github.io/qom-docs/v1.0.1) for the steps to install this libary.

### Running the Scripts

To run the scripts, navigate *inside* the top-level directory, and execute:

```bash
python scripts/bar/baz.py
```

Here, `bar` is the name of the folder (containing the version information) inside `scripts` and `baz.py` is the name of the script (refer to the repository structure).