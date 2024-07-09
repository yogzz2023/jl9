import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.Pp = np.eye(6)
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))
        self.Z1 = np.zeros((3, 1))
        self.Z2 = np.zeros((3, 1))
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = self.Z1[0]
            self.Sf[1] = self.Z1[1]
            self.Sf[2] = self.Z1[2]
            self.Meas_Time = time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.Sf[3] = vx
            self.Sf[4] = vy
            self.Sf[5] = vz
            self.Meas_Time = time
            print("Initialized filter state:")
            print("Sf:", self.Sf)
            print("Pf:", self.Pf)
            # Perform predict and update step after initializing from first two measurements
            self.predict_step(time)
            Z = np.array([[x], [y], [z]])
            self.update_step(Z)
            self.second_rep_flag = True

    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        self.Meas_Time = current_time
        print("Predicted filter state:")
        print("Sp:", self.Sp)
        print("Pp:", self.Pp)

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def gating(self, Z):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        d2 = np.dot(np.dot(np.transpose(Inn), np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az
    
    if az > 360:
        az = az - 360

    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))
         
        if x[i] > 0.0:                
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]       
        
        az[i] = az[i] * 180 / 3.14 

        if az[i] < 0.0:
            az[i] = 360 + az[i]
    
        if az[i] > 360:
            az[i] = az[i] - 360

    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)
            measurements.append((x, y, z, mt))
    return measurements

def form_measurement_groups(measurements, max_time_diff=0.50):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]
    
    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]
    
    if current_group:
        measurement_groups.append(current_group)
        
    return measurement_groups

# Create an instance of the Kalman filter
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_52_test.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Form measurement groups based on time
measurement_groups = form_measurement_groups(measurements)

csv_file_predicted = "ttk_52_test.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values

A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)

# Lists for plotting
r_list = []
az_list = []
el_list = []
time_list = []

# Initialize with the first two measurements
initialized = False

for group in measurement_groups:
    for i, (x, y, z, mt) in enumerate(group):
        if not kalman_filter.first_rep_flag:
            kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
        elif kalman_filter.first_rep_flag and not kalman_filter.second_rep_flag:
            if i > 0:
                x_prev, y_prev, z_prev = group[i-1][:3]
                dt = mt - group[i-1][3]
                vx = (x - x_prev) / dt
                vy = (y - y_prev) / dt
                vz = (z - z_prev) / dt
                kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
                initialized = True
            break
        elif not initialized:
            if i == 0:
                # If the first group has only one measurement, use the first measurement from the next group
                next_group = measurement_groups[1]
                if next_group:
                    x_next, y_next, z_next, mt_next = next_group[0]
                    dt = mt_next - mt
                    vx = (x_next - x) / dt
                    vy = (y_next - y) / dt
                    vz = (z_next - z) / dt
                    kalman_filter.initialize_filter_state(x_next, y_next, z_next, vx, vy, vz, mt_next)
                    initialized = True
            break
        else:
            # Start prediction and update steps from the third measurement onward
            if i >= 2 or (initialized and i > 0):
                Z = np.array([[x], [y], [z]])
                if kalman_filter.gating(Z):
                    kalman_filter.predict_step(mt)
                    kalman_filter.update_step(Z)

                # Convert to spherical coordinates for plotting
                r_val, az_val, el_val = cart2sph(kalman_filter.Sf[0], kalman_filter.Sf[1], kalman_filter.Sf[2])
                r_list.append(r_val)
                az_list.append(az_val)
                el_list.append(el_val)
                time_list.append(mt)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(time_list, r_list, color='b', label='Filtered Values')
plt.scatter(filtered_values_csv[:, 0], A[0], label='Filtered range (track id 31)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Range')
plt.title('Range over Time')
plt.legend()
mplcursors.cursor()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(time_list, az_list, color='b', label='Filtered Values')
plt.scatter(filtered_values_csv[:, 0], filtered_values_csv[:, 2], label='Filtered azimuth (track id 31)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Azimuth')
plt.title('Azimuth over Time')
plt.legend()
mplcursors.cursor()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(time_list, el_list, color='b', label='Filtered Values')
plt.scatter(filtered_values_csv[:, 0], A[2], label='Filtered elevation (track id 31)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Elevation')
plt.title('Elevation over Time')
plt.legend()
mplcursors.cursor()
plt.show()
