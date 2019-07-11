import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('ggplot')



column_names = ["train_loss","val_loss","epoch"]
experiment_name = "CE_central_patch_2_preprocessed"
results_path = os.path.join(os.path.dirname(__file__), os.pardir, "results")

summary_table = pd.
