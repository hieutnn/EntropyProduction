# Figures for "Hidden entropy production" (Paper 1)

Requirements: Python 3 with NumPy, SciPy, Matplotlib; a LaTeX install (pdflatex, TikZ) for Fig. 1.

## Fig. 1 (schematic)
    cd fig1 && pdflatex fig1.tex

## Fig. 2 (resolved, observed, hidden entropy production vs temperature)
    cd fig2
    python3 compute.py      # checks: Fock truncation, explicit reverse experiment, vacuum bounds
    python3 run_grid.py     # temperature sweep -> fig2_data.json (about 1 minute)
    python3 plot.py         # -> fig2.pdf, fig2.png

## Fig. 3 (observed entropy production at vacuum vs detector parameters)
    cd fig3
    python3 compute3.py     # Gaussian check of p*, sweeps over r, eta, theta -> fig3_data.json
    python3 plot3.py        # -> fig3.pdf, fig3.png
    python3 check_riccati_T.py   # finite-temperature check of the Gaussian equations (B10)

Parameters (Delta = gamma = 1, g = 0.6, gamma*delta = 1, M = 8, eta = 1, theta = 0.8, r = 0.05)
are set at the top of fig2/compute.py; fig3/model.py is a copy of fig2/compute.py.
The JSON files contain the data used for the published figures, so plot.py and plot3.py
run without recomputing.
