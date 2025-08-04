import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

output_path = Path('outputs/accuracy_on_ooc')
accuracy_paths = [output_path / Path(model + '_accuracy.csv') for model in ['vit', 'cnn', 'hybrid']]

model_names = []
accuracy_values = []
model_types = []
colors = []
for path in accuracy_paths:
    df = pd.read_csv(path)
    model_count = len(df)
    if path.name.startswith('vit'):
        model_types.extend(['Vision Transformer'] + ['_Vision Transformer'] * (model_count - 1))
        colors.extend(['#83c5be'] * model_count)
    elif path.name.startswith('cnn'):
        model_types.extend(['CNN'] + ['_CNN'] * (model_count - 1))
        colors.extend((['#e5989b'] * model_count))
    else:
        model_types.extend(['Hybrid'] + ['_Hybrid'] * (model_count - 1))
        colors.extend((['#457b9d'] * model_count))
    model_names.extend(df.iloc[:, 0].tolist())
    accuracy_values.extend([a*100 for a in df.iloc[:, 1].tolist()])

model_names = [name
               .replace('_patch', '/')
               .replace('small', 's')
               .replace('base', 'b')
               .replace('large', 'l')
               .replace('_224', '')
               .replace('_384', '') for name in model_names]

fig, ax = plt.subplots()
bar_container = ax.bar(model_names, accuracy_values, label=model_types, color=colors)
ax.set(title='Top-1-Accuracy on OOC dataset', xlabel='Model', ylabel='Accuracy (%)', ylim=(0, 100))
ax.bar_label(bar_container, fmt='{:,.2f}')
ax.set_xticklabels(model_names, rotation=30, ha='right')

ax.legend()
plt.show()