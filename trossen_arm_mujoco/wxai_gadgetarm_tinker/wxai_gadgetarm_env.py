import os
import numpy as np
import mujoco

class WxaiGadgetArmEnv:
    """
    Gymnasium-style MuJoCo Environment for WidowX 250 AI (WXAI) Robot Arm
    replicating the GadgetArm 3-Stage Reinforcement Learning Grasp Generation
    and Object Manipulation experiment (sensors-20-06183-v2.pdf) with tinker.obj.
    """

    def __init__(self, xml_path=None, max_steps=200, dt=0.04):
        if xml_path is None:
            xml_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "scene_tinker_gadgetarm.xml")
            )
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.arm_joint_names = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
        self.arm_joint_ids = [self.model.joint(name).id for name in self.arm_joint_names]
        self.arm_qpos_indices = [self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids]
        self.arm_dof_indices = [self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids]

        self.gripper_actuator_id = self.model.actuator("left_gripper").id
        self.left_finger_jid = self.model.joint("left_carriage_joint").id
        self.right_finger_jid = self.model.joint("right_carriage_joint").id
        self.left_finger_qadr = self.model.jnt_qposadr[self.left_finger_jid]
        self.right_finger_qadr = self.model.jnt_qposadr[self.right_finger_jid]
        self.left_finger_dofadr = self.model.jnt_dofadr[self.left_finger_jid]
        self.right_finger_dofadr = self.model.jnt_dofadr[self.right_finger_jid]

        self.ee_site_id = self.model.site("ee_site").id
        self.tinker_site_id = self.model.site("tinker_site").id
        self.target_site_id = self.model.site("target_site").id

        self.tinker_body_id = self.model.body("tinker_object").id
        self.table_body_id = self.model.body("table").id
        
        # Get body IDs for contact checking
        self.carriage_left_id = self.model.body("carriage_left").id
        self.carriage_right_id = self.model.body("carriage_right").id

        # Joint limits
        self.joint_min = np.array([self.model.jnt_range[jid][0] for jid in self.arm_joint_ids])
        self.joint_max = np.array([self.model.jnt_range[jid][1] for jid in self.arm_joint_ids])

        # Velocity scaling limits
        self.max_vel = np.array([2.5, 2.5, 2.5, 3.0, 3.0, 3.0])

        self.max_steps = max_steps
        self.dt = dt
        self.substeps = 5

        self.table_z = self.model.body("table").pos[2] + 0.025  # Table surface height (~0.0m)

        # Observation dimension: 38 dims total
        self.observation_dim = 38
        self.action_dim = 7  # 6 arm joints + 1 gripper command

        self.step_count = 0
        self.prev_dq = np.zeros(6)
        self.prev_d_eef_obj = 0.0
        self.prev_d_obj_target = 0.0
        self.prev_both_touch = False

    def get_ee_position(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_object_position(self):
        return self.data.site_xpos[self.tinker_site_id].copy()

    def get_target_position(self):
        return self.data.site_xpos[self.target_site_id].copy()

    def get_joint_positions(self):
        return np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices])

    def get_joint_velocities(self):
        return np.array([self.data.qvel[idx] for idx in self.arm_dof_indices])

    def get_object_velocity(self):
        body_dofadr = self.model.body_dofadr[self.tinker_body_id]
        return self.data.qvel[body_dofadr:body_dofadr+3].copy()

    def get_object_orientation(self):
        body_jntadr = self.model.body_jntadr[self.tinker_body_id]
        qpos_adr = self.model.jnt_qposadr[body_jntadr]
        return self.data.qpos[qpos_adr+3:qpos_adr+7].copy()

    def check_finger_contacts(self):
        """
        Check collision contacts between WXAI gripper fingers and tinker.obj
        """
        left_touch = False
        right_touch = False

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[con.geom1]
            geom2_body = self.model.geom_bodyid[con.geom2]

            bodies = {geom1_body, geom2_body}
            if self.tinker_body_id in bodies:
                if self.carriage_left_id in bodies:
                    left_touch = True
                if self.carriage_right_id in bodies:
                    right_touch = True

        return left_touch, right_touch

    def check_table_contacts(self):
        """
        Check collision contacts between WXAI arm/gripper and workspace table/floor
        """
        table_touch = False
        table_bodies = {self.table_body_id, 0}

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[con.geom1]
            geom2_body = self.model.geom_bodyid[con.geom2]

            if (geom1_body in table_bodies or geom2_body in table_bodies):
                if self.tinker_body_id not in {geom1_body, geom2_body}:
                    table_touch = True
                    break

        return table_touch

    def reset(self, start_q=None, randomize_obj=True):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        if start_q is None:
            # Initial joint posture bringing arm closer to workspace table
            start_q = np.array([0.28, 0.7, 0.5, 0.0, 0.3, 0.0])

        start_q = np.clip(start_q, self.joint_min, self.joint_max)
        for idx, val in zip(self.arm_qpos_indices, start_q):
            self.data.qpos[idx] = val
        for idx in self.arm_dof_indices:
            self.data.qvel[idx] = 0.0

        # Fully open gripper on reset (0.044)
        self.data.qpos[self.left_finger_qadr] = 0.044
        self.data.qpos[self.right_finger_qadr] = 0.044
        self.data.qvel[self.left_finger_dofadr] = 0.0
        self.data.qvel[self.right_finger_dofadr] = 0.0
        self.data.ctrl[self.gripper_actuator_id] = 0.044
        self.data.ctrl[:6] = start_q

        # Reset tinker object position resting on table surface (z = 0.0m)
        obj_jntadr = self.model.body_jntadr[self.tinker_body_id]
        obj_qpos_adr = self.model.jnt_qposadr[obj_jntadr]
        obj_dofadr = self.model.body_dofadr[self.tinker_body_id]
        
        base_obj_pos = np.array([0.35, 0.08, 0.0])
        if randomize_obj:
            noise = np.random.uniform(-0.015, 0.015, size=2)
            base_obj_pos[0] += noise[0]
            base_obj_pos[1] += noise[1]

        self.data.qpos[obj_qpos_adr:obj_qpos_adr+3] = base_obj_pos
        self.data.qpos[obj_qpos_adr+3:obj_qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)

        # Allow physics to settle object into a still position on table
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Zero object velocity after settling to ensure zero kinetic energy at episode start
        self.data.qvel[obj_dofadr:obj_dofadr+6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.prev_dq = np.zeros(6)
        self.prev_both_touch = False
        p_eef = self.get_ee_position()
        p_obj = self.get_object_position()
        p_target = self.get_target_position()

        target_ee_pos = np.array([p_obj[0], p_obj[1], self.table_z + 0.030])
        self.prev_d_target = np.linalg.norm(p_eef - target_ee_pos)
        self.prev_d_eef_obj = np.linalg.norm(p_obj - p_eef)
        self.prev_d_obj_target = np.linalg.norm(p_target - p_obj)

        return self._get_obs(), {}

    def _get_obs(self):
        q = self.get_joint_positions()
        p_eef = self.get_ee_position()
        dq = self.get_joint_velocities()
        p_obj = self.get_object_position()
        v_obj = self.get_object_velocity()
        quat_obj = self.get_object_orientation()
        p_target = self.get_target_position()

        d_eef_obj = np.linalg.norm(p_obj - p_eef)
        d_obj_target = np.linalg.norm(p_target - p_obj)

        vec_eef_obj = p_obj - p_eef
        vec_obj_target = p_target - p_obj

        left_touch, right_touch = self.check_finger_contacts()
        contacts = np.array([float(left_touch), float(right_touch)], dtype=np.float32)

        obs = np.concatenate([
            q, p_eef, dq, p_obj, v_obj, quat_obj, p_target,
            [d_eef_obj], [d_obj_target],
            vec_eef_obj, vec_obj_target, contacts
        ])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        arm_action = action[:6]
        gripper_action = action[6]

        curr_q = self.get_joint_positions()
        desired_dq = arm_action * self.max_vel
        desired_q = curr_q + desired_dq * self.dt
        desired_q = np.clip(desired_q, self.joint_min, self.joint_max)

        # Control 6 arm joint positions
        self.data.ctrl[:6] = desired_q

        # Smooth continuous gripper position control: action = -1 (open: 0.044), action = +1 (close: 0.00)
        target_gripper_pos = float(np.clip(0.044 * (1.0 - (gripper_action + 1.0) / 2.0), 0.00, 0.044))
        self.data.ctrl[self.gripper_actuator_id] = target_gripper_pos

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        new_q = self.get_joint_positions()
        new_p_eef = self.get_ee_position()
        new_p_obj = self.get_object_position()
        new_v_obj = self.get_object_velocity()
        new_p_target = self.get_target_position()
        new_dq = self.get_joint_velocities()

        ddq = (new_dq - self.prev_dq) / self.dt
        self.prev_dq = new_dq.copy()

        d_eef_obj = np.linalg.norm(new_p_obj - new_p_eef)
        d_obj_target = np.linalg.norm(new_p_target - new_p_obj)
        d_xy = np.linalg.norm(new_p_eef[:2] - new_p_obj[:2])
        ee_height = new_p_eef[2] - self.table_z

        left_touch, right_touch = self.check_finger_contacts()
        both_touch = left_touch and right_touch
        table_touch = self.check_table_contacts()
        too_low = (ee_height < 0.010)

        # -------------------------------------------------------------
        # Stage 1 Reward Formulation (Approach & Pre-Grasp Alignment):
        # -------------------------------------------------------------
        target_ee_pos = np.array([new_p_obj[0], new_p_obj[1], self.table_z + 0.030])
        d_target = np.linalg.norm(new_p_eef - target_ee_pos)

        # 1. Continuous Target Distance & Step-wise Progress Reward
        r_dist = -40.0 * d_target
        prev_d_target = getattr(self, "prev_d_target", d_target)
        r_prog = 400.0 * (prev_d_target - d_target)
        self.prev_d_target = d_target
        self.prev_d_eef_obj = d_eef_obj

        # 2. Gripper Open Penalty (Penalize closing during approach phase)
        gripper_open = (target_gripper_pos >= 0.036)
        r_open = 0.0 if gripper_open else -30.0

        # 3. Table Collision & Low Altitude Protection
        r_table = 0.0
        if table_touch:
            r_table -= 150.0
        elif ee_height < 0.010:
            r_table -= 50.0

        # 4. Object-Inside-Gripper Verification & Pre-Grasp Success Reward
        # Object is INSIDE open gripper when centered horizontally (d_xy <= 0.025m) and height is 0.015m <= z <= 0.045m
        object_inside_gripper = (d_xy <= 0.025) and (0.015 <= ee_height <= 0.045) and gripper_open and (not table_touch)
        
        r_inside = 0.0
        if object_inside_gripper:
            r_inside = 1000.0 + 500.0 * (1.0 - d_xy / 0.025)
        elif d_xy <= 0.06 and gripper_open and not table_touch:
            r_inside = 100.0 * (1.0 - d_xy / 0.06)

        # 5. Smoothness & Step Penalty
        r_smooth = -0.005 * np.sum(new_dq**2) - 0.0005 * np.sum(ddq**2)
        r_step = -1.0

        # 6. Drop / Knocked Penalty
        dropped = new_p_obj[2] < (self.table_z - 0.05)
        r_fail = -100.0 if dropped else 0.0

        reward = r_dist + r_prog + r_open + r_inside + r_table + r_smooth + r_step + r_fail

        # Stage 1 Success: Object perfectly inside open gripper WITHOUT touching the table
        success = object_inside_gripper
        done = success or dropped or (self.step_count >= self.max_steps)

        obs = self._get_obs()
        info = {
            "d_eef_obj": d_eef_obj,
            "d_target": d_target,
            "d_obj_target": d_obj_target,
            "d_xy": d_xy,
            "ee_height": ee_height,
            "both_touch": both_touch,
            "gripper_open": gripper_open,
            "table_touch": table_touch,
            "object_inside_gripper": object_inside_gripper,
            "success": success,
            "dropped": dropped,
            "r_dist": r_dist,
            "r_open": r_open,
            "r_inside": r_inside,
            "r_table": r_table,
            "r_smooth": r_smooth,
            "step_count": self.step_count
        }

        return obs, float(reward), done, False, info
