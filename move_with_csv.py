# Copyright 2025 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Purpose:
# This script demonstrates how to write a control loop to move the robot to different positions and
# record the states.

# Hardware setup:
# 1. A WXAI V0 arm with leader end effector and ip at 192.168.1.2

# The script does the following:
# 1. Initializes the driver
# 2. Configures the driver
# 3. Sets the robots to position mode
# 4. Records the sleep positions
# 5. Generates trajectory: sleep -> sleep -> home -> home -> sleep -> sleep
# 6. Moves the robot along the trajectory
# 7. Sets the robot to the idle mode
# 8. The driver automatically sets the mode to idle at the destructor

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
import csv

import trossen_arm

if __name__ == "__main__":
    print("Initializing the drivers...")
    driver = trossen_arm.TrossenArmDriver()

    print("Configuring the drivers...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        "192.168.1.2",
        False,
    )

    print("Moving to home positions...")
    driver.set_all_modes(trossen_arm.Mode.position)

    sleep_positions = np.array(driver.get_all_positions())
    home_positions = np.zeros(driver.get_num_joints())
    home_positions[1] = np.pi / 2
    home_positions[2] = np.pi / 2

    waypoints = np.array(
        [
            sleep_positions,
            sleep_positions,
            home_positions,
            home_positions,
            sleep_positions,
            sleep_positions,
        ]
    )
    timepoints = np.array([0, 1, 3, 4, 6, 7])

    interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)
    interpolator_feedforward_velocity = interpolator_position.derivative()
    interpolator_feedforward_acceleration = interpolator_feedforward_velocity.derivative()

    log_dict = {
        "time": [],
        "positions": [],
        "velocities": [],
        "efforts": [],
        "external_efforts": [],
    }

    start_time = time.time()
    end_time = start_time + timepoints[-1]

    while time.time() < end_time:
        loop_start_time = time.time()
        current_time = loop_start_time - start_time

        positions = interpolator_position(current_time)
        feedforward_velocity = interpolator_feedforward_velocity(current_time)
        feedforward_acceleration = interpolator_feedforward_acceleration(current_time)

        driver.set_all_positions(
            positions, 0.0, False, feedforward_velocity, feedforward_acceleration
        )

        log_dict["time"].append(current_time)
        log_dict["positions"].append(driver.get_all_positions())
        log_dict["velocities"].append(driver.get_all_velocities())
        log_dict["efforts"].append(driver.get_all_efforts())
        log_dict["external_efforts"].append(driver.get_all_external_efforts())

    plt.subplot(2, 2, 1)
    plt.plot(log_dict["time"], log_dict["positions"])
    plt.title("Positions")
    plt.subplot(2, 2, 2)
    plt.plot(log_dict["time"], log_dict["velocities"])
    plt.title("Velocities")
    plt.subplot(2, 2, 3)
    plt.plot(log_dict["time"], log_dict["efforts"])
    plt.title("Efforts")
    plt.subplot(2, 2, 4)
    plt.plot(log_dict["time"], log_dict["external_efforts"])
    plt.title("External Efforts")
    plt.savefig("move.png")

    csv_filename = "move_data.csv"
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time",
            "joint_0_position",
            "joint_1_position",
            "joint_2_position",
            "joint_3_position",
            "joint_4_position",
            "joint_5_position",
            "joint_6_position",
            "joint_0_velocity",
            "joint_1_velocity",
            "joint_2_velocity",
            "joint_3_velocity",
            "joint_4_velocity",
            "joint_5_velocity",
            "joint_6_velocity",
            "joint_0_effort",
            "joint_1_effort",
            "joint_2_effort",
            "joint_3_effort",
            "joint_4_effort",
            "joint_5_effort",
            "joint_6_effort",
            "joint_0_external_effort",
            "joint_1_external_effort",
            "joint_2_external_effort",
            "joint_3_external_effort",
            "joint_4_external_effort",
            "joint_5_external_effort",
            "joint_6_external_effort",
        ])

        for i in range(len(log_dict["time"])):
            positions = list(log_dict["positions"][i])
            velocities = list(log_dict["velocities"][i])
            efforts = list(log_dict["efforts"][i])
            external_effort = list(log_dict["external_efforts"][i])

            writer.writerow(
                positions
                + velocities
                + efforts
                + external_effort
            )
    print(f"Data saved to {csv_filename}")

