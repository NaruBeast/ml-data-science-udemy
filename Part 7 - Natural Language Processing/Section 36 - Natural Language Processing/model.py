import pandas as pd
import matplotlib as plt
import numpy as np

import re
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
