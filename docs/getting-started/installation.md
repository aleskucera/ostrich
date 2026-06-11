# Installation

This guide will walk you through installing Ostrich and its dependencies. The recommended method uses `uv`, a modern Python packaging tool that simplifies the entire process.

## Prerequisites

Before installing Ostrich, please ensure your system meets the following requirements.

### System and Software

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS.
- **Python**: Version 3.12 or higher.
- **CMake**: Version 3.x (specifically **below 4.0**) is required to build `openmesh`. CMake 4.x is not compatible.
- **CUDA** (Optional): Version 11.8 or higher is strongly recommended for significant performance gains on NVIDIA GPUs.

!!! tip "Windows Users: Use WSL2"
    While Ostrich can run on various systems, many core scientific computing libraries (including some used by Ostrich) have the best support and performance on Linux. We strongly recommend using the [Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/install) to create a Linux environment on your Windows machine.

### Environment Checks

1. **Verify your Python version**:

    ```bash
    python --version
    # Expected output: Python 3.12.x or higher
    ```

2. **Verify CMake version**:

    ```bash
    cmake --version
    # Expected output: cmake version 3.x.x (must be below 4.0)
    ```

    If your system CMake is version 4.x, download CMake 3.27 manually:

    ```bash
    # Download and extract CMake 3.27 to ~/.local/opt
    mkdir -p ~/.local/opt
    wget https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.tar.gz -P /tmp
    tar -xzf /tmp/cmake-3.27.0-linux-x86_64.tar.gz -C ~/.local/opt
    ```

    Then prepend it to your PATH for the installation step (see Step 4).

3. **Verify CUDA installation** (if you have an NVIDIA GPU):

    ```bash
    nvidia-smi
    # This command should output details about your GPU and the installed driver.
    ```

---

## Installation Steps

We use [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver. It replaces the need for `pip` and `venv` with a single, unified toolchain.

### Step 1: Install `uv`

If you don't have `uv` installed, choose the appropriate method for your OS. It's a standalone tool that can be installed quickly.

=== "Linux / macOS"

    ```bash
    # Use the official installer (recommended)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows (PowerShell)"

    ```powershell
    # Use the official PowerShell installer
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Step 2: Clone the Ostrich Repository

Get the source code from GitHub:

```bash
git clone https://github.com/aleskucera/ostrich.git
cd ostrich
```

### Step 3: Initialize Submodules

This step ensures all third-party dependencies included as git submodules are properly set up.

```bash
git submodule update --init --recursive
```

### Step 4: Create Environment and Install Dependencies

This is where `uv` shines. A single command handles everything.

=== "CMake 3.x (system default)"

    ```bash
    uv sync --extra sim
    ```

=== "CMake 4.x (need to use older CMake)"

    If your system CMake is 4.x, prepend the CMake 3.27 binary to PATH so `openmesh` builds correctly:

    ```bash
    PATH=~/.local/opt/cmake-3.27.0-linux-x86_64/bin:$PATH uv sync --extra sim
    ```

The `uv sync` command reads `pyproject.toml` and:

- **Creates a virtual environment** in the `.venv` directory.
- **Installs all required dependencies** at their exact pinned versions, ensuring a fully reproducible environment.

!!! info "What is `uv sync` doing?"
    Unlike traditional `pip install -r requirements.txt`, `uv sync` ensures that the environment is an *exact* reflection of the project's locked dependencies. It will add missing packages and remove ones that are not specified, guaranteeing a reproducible environment.

### Step 5: Verify Your Installation

To confirm everything is working, run one of the included examples.

```bash
uv run ball_bounce_example
```

The `uv run` command executes a command *inside* the virtual environment managed by `uv`. This saves you from having to manually run `source .venv/bin/activate` every time. The command `ball_bounce_example` is a shortcut defined in `pyproject.toml`.

---

## Dependency Overview

Ostrich relies on a set of high-performance libraries for physics, computation, and configuration. `uv sync` installs all of these for you.

| Package | Purpose |
| :--- | :--- |
| `warp-lang` | The core physics and graphics engine from NVIDIA that enables differentiable simulation. |
| `torch` | Provides the fundamental tensor library and autograd capabilities that make differentiation possible. |
| `hydra-core` | A powerful system for managing complex configurations in your simulations. |
| `h5py` | Used for efficient I/O, allowing you to log and save large datasets in the HDF5 format. |
| `trimesh` | A utility for loading and processing 3D mesh files for collision geometries. |
| `scipy` | Provides a broad set of scientific computing tools used across the library. |
| `nvtx` | Allows for deep performance profiling of code running on NVIDIA GPUs. |

---

!!! success "Installation Complete!"
    You are now ready to build and run your first simulation with Ostrich.

    -   Continue to the [**First Simulation Tutorial**](first-simulation.md) to build a simple physics scene.
    -   Explore the [**User Guide**](../user-guide/concepts.md) for in-depth explanations of advanced features.
    -   If you encounter any issues, please check the [**GitHub Issues**](https://github.com/aleskucera/ostrich/issues) page.
