from scipy.io import loadmat

# Load NASA B0005.mat file
data = loadmat('B0005.mat')
battery_data = data['B0005']['cycle'][0,0]
print("Dataset Loaded Successfully!")