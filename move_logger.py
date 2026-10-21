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
# Move the WXAI from sleep position to home position and back while recording
# extended robot parameters to a CSV file.

import csv
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

import trossen_arm


ARM_IP = "192.168.1.2"
CSV_FILENAME = "move_full_data.csv"
PLOT_FILENAME = "move_full_data.png"


if __name__ == "__main__":
    print("Initializing the driver...")
    driver = trossen_arm.TrossenArmDriver()

    print("Configuring the driver...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        ARM_IP,
        False,
    )

    num_joints = driver.get_num_joints()
    print(f"Connected. Number of joints: {num_joints}")

    print("Setting position mode...")
    driver.set_all_modes(trossen_arm.Mode.position)

    print("Reading sleep position...")
    sleep_positions = np.array(driver.get_all_positions())

    home_positions = np.zeros(num_joints)
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
        "accelerations": [],
        "efforts": [],
        "external_efforts": [],
        "compensation_efforts": [],
        "rotor_temperatures": [],
        "driver_temperatures": [],
        "cartesian_positions": [],
        "cartesian_velocities": [],
        "cartesian_accelerations": [],
        "error_information": [],
    }

    print("Starting motion and data logging...")
    start_time = time.time()
    end_time = start_time + timepoints[-1]

    while time.time() < end_time:
        loop_start_time = time.time()
        current_time = loop_start_time - start_time

        positions = interpolator_position(current_time)
        feedforward_velocity = interpolator_feedforward_velocity(current_time)
        feedforward_acceleration = interpolator_feedforward_acceleration(current_time)

        driver.set_all_positions(
            positions,
            0.0,
            False,
            feedforward_velocity,
            feedforward_acceleration,
        )

        log_dict["time"].append(current_time)
        log_dict["positions"].append(list(driver.get_all_positions()))
        log_dict["velocities"].append(list(driver.get_all_velocities()))
        log_dict["accelerations"].append(list(driver.get_all_accelerations()))
        log_dict["efforts"].append(list(driver.get_all_efforts()))
        log_dict["external_efforts"].append(list(driver.get_all_external_efforts()))
        log_dict["compensation_efforts"].append(
            list(driver.get_all_compensation_efforts())
        )
        log_dict["rotor_temperatures"].append(
            list(driver.get_all_rotor_temperatures())
        )
        log_dict["driver_temperatures"].append(
            list(driver.get_all_driver_temperatures())
        )
        log_dict["cartesian_positions"].append(
            list(driver.get_cartesian_positions())
        )
        log_dict["cartesian_velocities"].append(
            list(driver.get_cartesian_velocities())
        )
        log_dict["cartesian_accelerations"].append(
            list(driver.get_cartesian_accelerations())
        )
        log_dict["error_information"].append(
            driver.get_error_information()
        )

    print("Motion finished.")
    print("Saving CSV data...")

    csv_header = ["timestamp"]

    for joint_index in range(num_joints):
        csv_header.append(f"joint_{joint_index}_position")

    for joint_index in range(num_joints):
        csv_header.append(f"joint_{joint_index}_velocity")

    for joint_index in range(num_joints):
        csv_header.append(f"joint_{joint_index}_acceleration")

    for joint_index in range(num_joints):
        csv_header.append(f"joint_{joint_index}_effort")

    for joint_index in range(num_joints):
        csv_header.append(f"joint_{joint_index}_external_effort")

    for joint_index in range(num_joints):
        csv_header.append(f"joint_{joint_index}_compensation_effort")

    for joint_index in range(num_joints):
        csv_header.append(f"joint_{joint_index}_rotor_temperature")

    for joint_index in range(num_joints):
        csv_header.append(f"joint_{joint_index}_driver_temperature")

    csv_header.extend(
        [
            "cartesian_x",
            "cartesian_y",
            "cartesian_z",
            "cartesian_rx",
            "cartesian_ry",
            "cartesian_rz",
            "cartesian_vx",
            "cartesian_vy",
            "cartesian_vz",
            "cartesian_wx",
            "cartesian_wy",
            "cartesian_wz",
            "cartesian_ax",
            "cartesian_ay",
            "cartesian_az",
            "cartesian_alphax",
            "cartesian_alphay",
            "cartesian_alphaz",
            "error_information",
        ]
    )

    with open(CSV_FILENAME, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

        for row_index in range(len(log_dict["time"])):
            positions = log_dict["positions"][row_index]
            velocities = log_dict["velocities"][row_index]
            accelerations = log_dict["accelerations"][row_index]
            efforts = log_dict["efforts"][row_index]
            external_efforts = log_dict["external_efforts"][row_index]
            compensation_efforts = log_dict["compensation_efforts"][row_index]
            rotor_temperatures = log_dict["rotor_temperatures"][row_index]
            driver_temperatures = log_dict["driver_temperatures"][row_index]
            cartesian_positions = log_dict["cartesian_positions"][row_index]
            cartesian_velocities = log_dict["cartesian_velocities"][row_index]
            cartesian_accelerations = log_dict["cartesian_accelerations"][row_index]
            error_information = log_dict["error_information"][row_index]

            writer.writerow(
                [log_dict["time"][row_index]]
                + positions
                + velocities
                + accelerations
                + efforts
                + external_efforts
                + compensation_efforts
                + rotor_temperatures
                + driver_temperatures
                + cartesian_positions
                + cartesian_velocities
                + cartesian_accelerations
                + [error_information]
            )

    print(f"CSV saved: {CSV_FILENAME}")

    print("Creating plot...")
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 2, 1)
    plt.plot(log_dict["time"], log_dict["positions"])
    plt.title("Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")

    plt.subplot(3, 2, 2)
    plt.plot(log_dict["time"], log_dict["velocities"])
    plt.title("Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(3, 2, 3)
    plt.plot(log_dict["time"], log_dict["accelerations"])
    plt.title("Joint Accelerations")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/s^2)")

    plt.subplot(3, 2, 4)
    plt.plot(log_dict["time"], log_dict["efforts"])
    plt.title("Joint Efforts")
    plt.xlabel("Time (s)")
    plt.ylabel("Effort")

    plt.subplot(3, 2, 5)
    plt.plot(log_dict["time"], log_dict["rotor_temperatures"])
    plt.title("Rotor Temperatures")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")

    plt.subplot(3, 2, 6)
    plt.plot(log_dict["time"], log_dict["driver_temperatures"])
    plt.title("Driver Temperatures")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")

    plt.tight_layout()
    plt.savefig(PLOT_FILENAME)

    print(f"Plot saved: {PLOT_FILENAME}")
    print("Done.")
