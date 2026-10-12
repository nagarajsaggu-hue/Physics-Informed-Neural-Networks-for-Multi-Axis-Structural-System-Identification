# Phase 1 - PINN System Identification (Multi-Axis)

**Status: DIRECTION-AGNOSTIC ARCHITECTURE COMPLETE.** 

## What this project does

A steel table is shaken by a horizontal harmonic force in ANSYS. The displacement of one corner node is recorded across three spatial directions (X, Y, Z). A physics-informed neural network is trained to reproduce that displacement while simultaneously identifying the stiffness and damping of an equivalent single-degree-of-freedom system for the selected axis.

The codebase is completely direction-agnostic; axis selection is governed entirely by `config.yaml`.

## Key Learnings & Physics Constraints
* **Primary Axes (X, Y):** The 1-DOF PINN successfully extracts highly accurate stiffness and damping parameters because the horizontal external force directly drives these directions.
* **Parasitic Axes (Z):** The 1-DOF mathematical formulation collapses on the Z-axis. The external horizontal force (11,103 N) does not act in the Z-direction; the microscopic Z-displacement is merely a coupled structural vibration. Forcing a 1-DOF equation to balance a massive applied force against a parasitic vibration results in invalid parameters (negative least-squares stiffness, absurd PINN damping). This is a physical constraint of the 1-DOF equation, not a pipeline failure.

## Conventions

- **Direction Switching:** One configuration block (X, Y, or Z) must be active in `config.yaml` at a time. The active block strictly dictates the generalized mass, observation column, and output subdirectory.
- Submit a job, end the turn, come back later. Do not poll `squeue` in a loop.
- Do not edit files in `data/` - they come from ANSYS and are regenerated there.
