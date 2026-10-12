# Phase 1 - PINN System Identification

Identifying the stiffness and damping of a steel table from a single node's
displacement response across multiple axes (X, Y, Z), using a physics-informed neural network.

## Quick start

    source venv/bin/activate
    which python                          # must end in venv/bin/python

    # Select the target axis (X, Y, or Z) by uncommenting its block in config.yaml
    python -m pytest tests/ -v            
    python scripts/run_train.py --tag phase1_run
    python scripts/run_evaluate.py --tag phase1_run
    python scripts/run_plot.py --tag phase1_run

## Multi-Axis Flexibility

The model is direction-agnostic. The `config.yaml` file contains mutually exclusive blocks for the X, Y, and Z directions. 
* To switch directions, comment out the inactive blocks and uncomment the desired one. 
* Output JSONs, CSVs, and figures are automatically routed to specific subdirectories to strictly prevent data overwriting between cluster jobs.
* The Z-axis data utilizes a 500 Hz sampling rate (5000 samples), requiring a higher `num_domain` collocation density in the configuration.
