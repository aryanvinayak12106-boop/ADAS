#Write a program to simulate an Adaptive Cruise Control (ACC) system that adjusts the vehicle speed according to the distance from the vehicle ahead.
import numpy as np

# ACC values
desired_distance = 40       # m
actual_distance = 30        # m
our_speed = 80               # km/h
lead_speed = 60              # km/h

# PID gains
Kp = 0.5
Ki = 0.1
Kd = 0.2

# PID assumptions
accumulated_error = 5
rate_of_error = 2

# Distance error
error = desired_distance - actual_distance

# PID calculations
P = Kp * error
I = Ki * accumulated_error
D = Kd * rate_of_error

# Total PID output
PID_output = P + I + D

# Adjust vehicle speed
our_speed = our_speed - PID_output

print("Distance Error =", error, "m")
print("P =", P)
print("I =", I)
print("D =", D)
print("PID Output =", PID_output)
print("Lead Vehicle Speed =", lead_speed, "km/h")
print("Our New Speed =", our_speed, "km/h")