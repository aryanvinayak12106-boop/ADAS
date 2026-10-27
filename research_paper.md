# Real-Time Vehicle Detection and Time-to-Collision Estimation in a Procedural Driving Simulator

## Abstract

Advanced driver-assistance systems (ADAS) must detect nearby vehicles and provide timely warnings without overwhelming the driver with false alarms. This work presents a customized vehicle-awareness module implemented in the MetaDrive procedural driving simulator. The system uses a 240-ray Lidar configured for a 20 m detection range, identifies vehicles in the ego vehicle's forward field, and estimates time to collision (TTC) from relative distance and closing speed. A red visual warning is displayed when a vehicle is detected within 20 m ahead, while a live TTC value is shown for the most imminent approaching vehicle. The ego vehicle is rendered as a black L-class vehicle, and vehicle-collision termination is disabled to support continued interactive testing after an impact. The implementation was syntax-validated and executed successfully in MetaDrive's interactive and headless environments. The current work is a simulation prototype; quantitative detection accuracy, false-positive rate, and safety improvement require controlled experiments with recorded ground truth.

**Keywords:** advanced driver-assistance systems, vehicle detection, Lidar, time to collision, MetaDrive, driving simulation, collision warning

## 1. Introduction

Vehicle collision warnings are an important component of ADAS. A useful warning system must answer two related questions:

1. Is another vehicle present within a relevant region around the ego vehicle?
2. If the vehicles are approaching one another, how much time remains before a collision?

Distance alone is insufficient because a distant vehicle traveling away from the ego vehicle may be safer than a nearby vehicle closing rapidly. Time to collision combines separation and relative motion into a compact risk indicator:

$$
\mathrm{TTC} = \frac{d}{v_c},
$$

where $d$ is the distance between vehicles and $v_c$ is the positive closing speed.

This project develops and evaluates a simulation prototype that integrates vehicle detection, forward warning visualization, and TTC estimation into MetaDrive. The simulator provides controllable traffic, repeatable scenarios, rendering, Lidar observations, and collision state information. These properties make it suitable for early-stage development before testing on physical sensors or a real vehicle.

The contributions of this work are:

- A configurable Lidar-based vehicle detection pipeline in MetaDrive.
- A forward-region warning for vehicles within a configurable distance.
- A live TTC estimate based on relative position and velocity.
- A visual red warning overlay for immediate driver feedback.
- An interactive test configuration using a black L-class ego vehicle.
- A documented basis for future quantitative ADAS experiments.

## 2. System Overview

The implementation is based on the MetaDrive simulator, an open-source procedural driving environment for autonomous-driving and reinforcement-learning research. The customized interactive entry point is:

`metadrive/examples/drive_in_single_agent_env.py`

The system consists of the following stages:

1. MetaDrive generates a road scene and traffic.
2. The ego vehicle receives Lidar-based surrounding-object information.
3. Dynamic objects are filtered to retain vehicles.
4. Vehicle positions are converted into the ego vehicle's local coordinate frame.
5. Vehicles within the configured forward range trigger a warning.
6. Relative position and velocity are used to estimate TTC.
7. TTC and warning status are shown in the simulator interface.

The current interactive configuration is:

| Parameter | Value |
|---|---:|
| Ego vehicle model | L |
| Ego vehicle color | Black |
| Lidar rays | 240 |
| Lidar range | 20 m |
| Traffic density | 0.1 |
| Rendering | Enabled |
| Manual control | Enabled |
| Vehicle collision termination | Disabled |

The 20 m value applies to the customized interactive simulation. The base MetaDrive configuration retains a 75 m default Lidar range, while other MetaDrive examples may override that value.

## 3. Vehicle Detection

### 3.1 Lidar configuration

The ego vehicle uses 240 Lidar rays distributed around the vehicle. The interactive example configures the Lidar range as 20 m:

```python
lidar=dict(num_lasers=240, distance=20)
```

The Lidar provides a broad-phase set of nearby objects. The implementation filters this set using MetaDrive's `BaseVehicle` type so that static scenery and non-vehicle objects are not treated as vehicle warnings.

### 3.2 Forward-region classification

For each detected vehicle, its world position is converted to the ego vehicle's local coordinate frame. The first local coordinate represents the forward direction of the ego vehicle. A vehicle is classified as being ahead when:

$$
0 < x_\mathrm{local} \leq 20 \text{ m}.
$$

When this condition is met, the simulator displays the red message:

```text
VEHICLE AHEAD WITHIN 20 m
```

The warning is dynamic: it is cleared when no detected vehicle satisfies the forward-region condition.

This implementation detects vehicles around the ego vehicle for TTC processing, but the red warning is specifically limited to the forward region. Vehicles behind or beside the ego vehicle can therefore influence TTC when they are closing, but they do not trigger the forward warning text.

## 4. Time-to-Collision Estimation

### 4.1 Relative position

For every detected vehicle, the implementation computes the planar relative position:

$$
\mathbf{r} = \mathbf{p}_\mathrm{other} - \mathbf{p}_\mathrm{ego}.
$$

The separation distance is:

$$
d = \|\mathbf{r}\|.
$$

Only the horizontal plane is used because the driving scenario is road-based.

### 4.2 Closing speed

The relative velocity used by the prototype is:

$$
\mathbf{v}_r = \mathbf{v}_\mathrm{ego} - \mathbf{v}_\mathrm{other}.
$$

The unit vector from the ego vehicle to the other vehicle is:

$$
\hat{\mathbf{r}} = \frac{\mathbf{r}}{\|\mathbf{r}\|}.
$$

The scalar closing speed is:

$$
v_c = \mathbf{v}_r \cdot \hat{\mathbf{r}}.
$$

Only $v_c > 0$ is considered an approaching condition. For such vehicles:

$$
\mathrm{TTC} = \frac{d}{v_c}.
$$

The simulator displays the smallest finite TTC among all approaching vehicles. If no approaching vehicle is detected, it displays:

```text
No imminent collision
```

### 4.3 Interpretation

The TTC estimate is a kinematic approximation. It assumes that the current relative velocity remains approximately constant and does not explicitly model:

- Braking or acceleration changes.
- Steering changes.
- Vehicle dimensions and collision envelopes.
- Lane changes.
- Occlusion.
- Sensor noise or missed detections.
- Road curvature over the prediction interval.

Consequently, the displayed TTC should be interpreted as a warning indicator rather than an exact physical collision time.

## 5. Reward and Environment Configuration

The underlying MetaDrive environment defines the following default reward terms:

| Component | Value |
|---|---:|
| Destination success reward | +10.0 |
| Driving reward | +1.0 |
| Speed reward | +0.1 |
| Off-road penalty | -5.0 |
| Vehicle collision penalty | -5.0 |
| Object collision penalty | -5.0 |
| Sidewalk collision penalty | 0.0 |

The default costs include:

| Cost | Value |
|---|---:|
| Vehicle collision cost | 1.0 |
| Object collision cost | 1.0 |
| Off-road cost | 1.0 |

For interactive testing, vehicle-collision termination is disabled:

```python
crash_vehicle_done = False
```

This allows the user to continue driving after an impact and inspect the warning behavior. It is not intended to represent a production safety policy.

## 6. PPO and Control Context

The repository includes a pre-trained NumPy PPO expert, but the customized interactive example is primarily a manual-control demonstration. The available expert model uses:

- A 275-dimensional observation.
- Two fully connected hidden layers of 256 units each.
- Tanh activations.
- Two action means and two log-standard-deviation outputs.
- A separate value network with two 256-unit hidden layers and one scalar output.

The expert observation configuration uses a 50 m Lidar range, four other-vehicle slots, and no active side or lane-line detector lasers. This differs from the customized interactive warning configuration, which uses a 20 m Lidar range and performs its own surrounding-vehicle scan. The repository does not define a new PPO training run or provide project-specific learning-rate, batch-size, discount-factor, GAE, or clipping settings. Therefore, this work should be described as a perception and warning prototype rather than a newly trained PPO policy.

## 7. Experimental Procedure

### 7.1 Installation and execution

The project was installed in editable mode using Python 3.11.9:

```powershell
python -m pip install -e .
```

The interactive simulation is launched with:

```powershell
cd "C:\Users\Aryan salunkhe\OneDrive\Desktop\ADAS"
python -m metadrive.examples.drive_in_single_agent_env
```

### 7.2 Controls

| Key | Function |
|---|---|
| W | Accelerate |
| S | Brake |
| A | Steer left |
| D | Steer right |
| R | Reset environment |
| B | Top-down camera |
| Q | Third-person camera |
| H | Display help |
| Esc | Quit |

### 7.3 Current verification

The implementation has been verified through:

- Python syntax compilation of the modified modules.
- MetaDrive environment initialization.
- Effective vehicle configuration checks.
- Headless MetaDrive simulation execution.
- Interactive onscreen simulation startup.

A 100-step headless MetaDrive profile completed successfully during development. This verifies simulator execution, but it is not a statistical evaluation of detection performance.

## 8. Results and Discussion

The current prototype demonstrates end-to-end integration of:

1. Procedural traffic simulation.
2. Lidar-based surrounding-vehicle retrieval.
3. Forward vehicle classification.
4. Relative-kinematics TTC estimation.
5. Real-time red warning visualization.

The warning distance is deterministic in the configured simulation: a vehicle outside 20 m is not included in the customized scan, while a vehicle within the forward local-coordinate region can trigger the warning. TTC is continuously recomputed each simulation step and uses the most imminent approaching vehicle.

However, the current evidence is functional rather than quantitative. No labeled dataset, repeated scenario benchmark, precision/recall analysis, latency measurement, or comparison against a baseline detector has yet been performed. The simulator's collision state should not be treated as a direct measure of detection accuracy because a collision event and a sensor detection are different outcomes.

## 9. Limitations

The prototype has several limitations:

1. **Point-center distance:** TTC uses vehicle-center separation rather than the distance between collision envelopes.
2. **Constant-velocity assumption:** The estimate does not predict braking, acceleration, or steering.
3. **No uncertainty model:** Lidar noise, dropout, occlusion, and multipath effects are not modeled in the warning logic.
4. **No warning hysteresis:** The forward warning can change as vehicles enter or leave the 20 m region.
5. **No severity thresholds:** The interface does not currently classify TTC into caution, warning, and emergency levels.
6. **No independent ground truth:** Detection results have not been compared against known vehicle positions in a controlled benchmark.
7. **No physical validation:** Results from MetaDrive do not establish performance for automotive radar, camera, or production Lidar hardware.
8. **PPO mismatch:** The included PPO expert uses a different observation configuration from the customized warning module.

## 10. Future Work

Future development should include:

- A scenario generator with known vehicle trajectories and collision envelopes.
- Precision, recall, false-positive, and false-negative measurements.
- TTC error analysis against ground-truth collision times.
- Separate front, rear, and blind-spot warning zones.
- Vehicle-width and length-aware gap calculations.
- TTC smoothing and warning hysteresis.
- Warning severity levels based on TTC thresholds.
- Sensor-noise and dropout experiments.
- Evaluation across traffic density, road curvature, speed, and weather conditions.
- A reproducible PPO training configuration with documented hyperparameters.
- Comparison with a baseline distance-only warning system.
- Hardware-in-the-loop tests using automotive radar or camera data.

## 11. Conclusion

This work presents a functional simulation prototype for vehicle detection and time-to-collision warning in MetaDrive. The customized system uses a 240-ray, 20 m Lidar configuration, identifies vehicles in the ego vehicle's forward region, estimates TTC from relative distance and closing speed, and displays a red warning when a vehicle is within 20 m ahead. The black L-class ego vehicle and interactive rendering configuration make the behavior easy to inspect during manual driving.

The implementation establishes the software foundation for a more rigorous ADAS study. The next step is not to claim real-world detection performance, but to construct repeatable labeled scenarios and measure warning accuracy, TTC error, latency, and safety outcomes under controlled conditions.

## References

1. Q. Li, Z. Peng, L. Feng, Q. Zhang, Z. Xue, and B. Zhou, “MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2022.
2. MetaDrive documentation and source repository: https://github.com/metadriverse/metadrive
3. P. J. G. Ribeiro, C. M. M. Silva, and others, “Time-to-collision estimation and its application in vehicle safety systems,” related ADAS literature. A specific publication should be selected and cited during formal submission.

