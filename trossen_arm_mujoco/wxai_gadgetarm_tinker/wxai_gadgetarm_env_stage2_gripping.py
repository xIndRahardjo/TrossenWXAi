import os
import numpy as np
import mujoco

class WxaiGadgetArmStage2Env:
    """
    Gymnasium-style MuJoCo Environment for Stage 2 (Gripping Stage)
    of the decoupled GadgetArm experiment (sensors-20-06183-v2.pdf).
    
    In Stage 2:
    - The WXAI arm starts pre-positioned directly around tinker.obj with fingers wide OPEN.
    - Arm posture is anchored to pre-grasp posture (with micro-adjustments allowed).
    - Focuses strictly on learning finger closure, grip force, and maintaining 100% dual-finger contact.
    """

    def __init__(self, xml_path=None, max_steps=100, dt=0.04):
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
        
        self.carriage_left_id = self.model.body("carriage_left").id
        self.carriage_right_id = self.model.body("carriage_right").id

        # Joint limits
        self.joint_min = np.array([self.model.jnt_range[jid][0] for jid in self.arm_joint_ids])
        self.joint_max = np.array([self.model.jnt_range[jid][1] for jid in self.arm_joint_ids])

        self.pre_grasp_q = np.array([0.393, 1.455, 0.494, 0.213, 0.226, 0.0])

        self.max_steps = max_steps
        self.dt = dt
        self.substeps = 5

        self.table_z = self.model.body("table").pos[2] + 0.025

        self.observation_dim = 38
        self.action_dim = 7  # 6 micro-adjustments + 1 gripper close action

        self.step_count = 0
        self.prev_dq = np.zeros(6)
        self.prev_d_eef_obj = 0.0
        self.prev_both_touch = False
        self.consecutive_touch_count = 0
        self.initial_obj_pos = np.zeros(3)

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

    def reset(self, randomize_obj=False):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.consecutive_touch_count = 0

        # Stage 2 Pre-Grasp Pose: WXAI arm pre-aligned right at tinker.obj
        for idx, val in zip(self.arm_qpos_indices, self.pre_grasp_q):
            self.data.qpos[idx] = val
        for idx in self.arm_dof_indices:
            self.data.qvel[idx] = 0.0

        # Fully open gripper on Stage 2 reset (0.044)
        self.data.qpos[self.left_finger_qadr] = 0.044
        self.data.qpos[self.right_finger_qadr] = 0.044
        self.data.qvel[self.left_finger_dofadr] = 0.0
        self.data.qvel[self.right_finger_dofadr] = 0.0
        self.data.ctrl[self.gripper_actuator_id] = 0.044
        self.data.ctrl[:6] = self.pre_grasp_q

        # Fixed tinker object position resting on table surface (z = 0.0m)
        obj_jntadr = self.model.body_jntadr[self.tinker_body_id]
        obj_qpos_adr = self.model.jnt_qposadr[obj_jntadr]
        obj_dofadr = self.model.body_dofadr[self.tinker_body_id]
        
        base_obj_pos = np.array([0.35, 0.10, 0.0])
        if randomize_obj:
            noise = np.random.uniform(-0.003, 0.003, size=2)
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

        # Record initial settled object position
        self.initial_obj_pos = self.get_object_position().copy()

        self.prev_dq = np.zeros(6)
        self.prev_both_touch = False
        p_eef = self.get_ee_position()
        p_obj = self.get_object_position()

        self.prev_d_eef_obj = np.linalg.norm(p_obj - p_eef)

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
        micro_arm_action = action[:6]
        gripper_action = action[6]

        # Stage 2 Arm Posture Control:
        # Pre-grasp posture when searching for grip; once dual contact is established, smoothly blend upward lifting posture
        curr_q = self.get_joint_positions()
        desired_dq = micro_arm_action * np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3])
        
        left_touch, right_touch = self.check_finger_contacts()
        both_touch = left_touch and right_touch

        if both_touch or self.consecutive_touch_count > 0:
            lift_progress = min(1.0, (self.consecutive_touch_count + 1) / 15.0)
            target_lift = np.array([0.0, -0.15, 0.20, 0.10, 0.0, 0.0]) * lift_progress
            target_arm_q = self.pre_grasp_q + target_lift + desired_dq * self.dt
        else:
            target_arm_q = self.pre_grasp_q + desired_dq * self.dt

        target_arm_q = np.clip(target_arm_q, self.joint_min, self.joint_max)
        self.data.ctrl[:6] = target_arm_q

        # Gripper closing control: action > 0 closes fingers towards 0.00
        target_gripper_pos = float(np.clip(0.044 * (1.0 - (gripper_action + 1.0) / 2.0), 0.00, 0.044))
        self.data.ctrl[self.gripper_actuator_id] = target_gripper_pos

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        new_q = self.get_joint_positions()
        new_p_eef = self.get_ee_position()
        new_p_obj = self.get_object_position()
        new_v_obj = self.get_object_velocity()
        new_dq = self.get_joint_velocities()

        ddq = (new_dq - self.prev_dq) / self.dt
        self.prev_dq = new_dq.copy()

        d_eef_obj = np.linalg.norm(new_p_obj - new_p_eef)
        obj_disp = np.linalg.norm(new_p_obj[:2] - self.initial_obj_pos[:2])
        obj_lift = new_p_obj[2] - self.table_z

        left_touch, right_touch = self.check_finger_contacts()
        both_touch = left_touch and right_touch
        either_touch = left_touch or right_touch

        if both_touch:
            self.consecutive_touch_count += 1
        else:
            self.consecutive_touch_count = 0

        # -------------------------------------------------------------
        # 3-Stage Reward Formulation (sensors-20-06183 Section 4.3):
        # 1. Alignment Stage (Gripper movement towards object center)
        # 2. Gripping Stage (Dual finger contact)
        # 3. Uplift & Gravity Stage (Lifting object off table surface)
        # -------------------------------------------------------------
        # 1. Gripping Stage Reward
        r_grasp = 0.0
        if both_touch and not self.prev_both_touch:
            r_grasp = 200.0  # Initial dual contact transition bonus
        elif both_touch:
            r_grasp = 100.0  # Dual contact maintenance reward
        elif either_touch:
            r_grasp = 15.0
        self.prev_both_touch = both_touch

        # 2. Uplifting Stage Reward (Reward lifting tinker.obj off table surface)
        r_lift = 0.0
        if both_touch and obj_lift > 0.005:
            r_lift = 500.0 * obj_lift + (200.0 if obj_lift > 0.025 else 0.0)

        # 3. Object Drop / Instability Penalty
        r_drop = 0.0
        if not both_touch and self.prev_both_touch and obj_lift < 0.01:
            r_drop = -150.0  # Penalty for dropping object after grip

        r_stability = -50.0 * np.linalg.norm(new_v_obj) - 30.0 * obj_disp
        r_smooth = -0.005 * np.sum(new_dq**2) - 0.0005 * np.sum(ddq**2)

        reward = r_grasp + r_lift + r_drop + r_stability + r_smooth

        # Stage 2 Success Condition: Dual contact maintained AND object uplifted off table (>= 2.5cm)
        is_lifted = (obj_lift >= 0.025)
        success = both_touch and is_lifted and (self.consecutive_touch_count >= 10)
        done = success or (self.step_count >= self.max_steps)

        obs = self._get_obs()
        info = {
            "d_eef_obj": d_eef_obj,
            "both_touch": both_touch,
            "consecutive_touch_count": self.consecutive_touch_count,
            "obj_disp": obj_disp,
            "obj_lift": obj_lift,
            "is_lifted": is_lifted,
            "success": success,
            "r_grasp": r_grasp,
            "r_lift": r_lift,
            "r_drop": r_drop,
            "r_stability": r_stability,
            "r_smooth": r_smooth,
            "step_count": self.step_count
        }

        return obs, float(reward), done, False, info
