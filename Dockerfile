FROM figs:latest

# SINGER-specific dependencies (not in the FiGS base image)
RUN python -m pip install --no-cache-dir \
    typer \
    wandb \
    "transformers<4.37" \
    plotly \
    seaborn \
    pandas \
    onnxruntime

# Re-pin numpy after all installs (downstream packages may pull in numpy 2.x)
RUN python -m pip install --no-cache-dir numpy==1.26.4
