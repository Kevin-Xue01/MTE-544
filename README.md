# MTE546 Final Project
This is the code used to collect data and genearte the results for the final project for for MTE546. Below is a guide for how to use the code and re-produce our results.

---

## Collecting Data
To collect data, you'll need an environment which can run ROS2 and Gazebo. Setup instruction from the MTE544 course can be found in `vm_guides/` for Windows and AppleSilicon systems. Before running the robot software, you'll also likely need to create a virtual environment. Your can create a virtual environment by running,
- `python3 -m venv my_venv`
- `source my_venv/bin/activate`
- `pip install -r requirements.txt`

Once you have your environment setup, data can be collected by,
1. Setting the correct `localization_mode`, `path_type` and `training_iteration` in `utils/config.py`
    - Values for each are defined in `utils/constants.py`
    - `training_iteration` controls what folder data is written to in `csv/`
2. Start the Gazebo simulation
    - Open a terminal
    - Run `export TURTLEBOT3_MODEL=burger`
    - Run `ros2 launch turtlebot3_gazebo empty_world.launch.py`
3. Start the robot software
    - Open another terminal
    - source the venv you created previously
    - Run `python3 decisions.py`

You should see the robot in simulation start to move according to the programmed trajectory.

### Plotting Simulation Data
To plot simulation data,
1. Set the `training_iteration` and `localization_mode` you would like to plot in `utils/configs.py`
2. Open a terminal
3. Source your virtual environment
4. Run `python3 plot_trajectories.py`
This will create plots inside of the trajectory folders of the `training_iteration`.

---

## Tuning Parameters
The tuning scripts for for the EKF and UKF are `tune_kf.py` and `tune_ekf.py`. To run the tuning process,
1. Set the `training_iteration` and `localization_mode` you would like to tune in `utils/configs.py`
2. Within the tuning scripts, inside the `optimize_noise_params` function, set the Q and R diagonal values you want the tuning process to start from.
3. Open a terminal
4. Source your virtual environment
5. Run `python3 tune_<>.py`
The script may take several minutes to return. You should see the final Q and R values printed in the terminal, and plots showing the before and after training results in the corresponding `training_iteration` folders.
