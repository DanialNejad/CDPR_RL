import pandas as pd
import numpy as np
from scipy.optimize import linprog

# تعریف داده‌های ورودی و خروجی
data = {
    'DMU': ['DMU1', 'DMU2', 'DMU3', 'DMU4', 'DMU5'],
    'x1': [10, 12, 8, 11, 9],
    'x2': [500, 600, 450, 550, 480],
    'x3': [15, 18, 12, 16, 14],
    'y1': [1000, 1100, 950, 1050, 980],
    'y2': [800, 900, 750, 850, 770],
    'y3': [90, 85, 88, 92, 87]
}

# تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# آماده‌سازی داده‌ها برای DEA
inputs = df[['x1', 'x2', 'x3']].values
outputs = df[['y1', 'y2', 'y3']].values
dmu_names = df['DMU'].values

def dea_ccr_input(inputs, outputs):
    num_dmus, num_inputs = inputs.shape
    num_outputs = outputs.shape[1]
    
    efficiencies = []
    
    for i in range(num_dmus):
        c = np.hstack([np.zeros(num_inputs), -1])
        
        A = np.hstack([inputs, np.ones((num_dmus, 1))])
        b = inputs[i, :]
        
        G = np.hstack([-outputs, np.zeros((num_dmus, 1))])
        h = -outputs[i, :]
        
        bounds = [(0, None) for _ in range(num_inputs)] + [(None, None)]
        
        res = linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=b, bounds=bounds, method='highs')
        
        efficiency = 1 / res.x[-1]
        efficiencies.append(efficiency)
    
    return efficiencies

# محاسبه کارایی
efficiencies = dea_ccr_input(inputs, outputs)

# نمایش نتایج
for dmu, efficiency in zip(dmu_names, efficiencies):
    print(f'{dmu}: {efficiency:.2f}')
