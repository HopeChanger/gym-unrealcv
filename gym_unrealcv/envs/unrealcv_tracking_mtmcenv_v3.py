import math
import gym
import random

import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking.baseline import *
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.tracking.interaction import Tracking
import gym_unrealcv
import cv2
import matplotlib.pyplot as plt
from gym_unrealcv.envs.utils.misc import *

from mmtrack.apis import inference_mot, init_model

# from sacred import SETTINGS
# SETTINGS['CAPTURE_MODE'] = "no"

''' 
It is an env for multi-camera active object tracking.
State : raw color image
Action:  rotate cameras (yaw, pitch)
Task: Learn to follow the target object(moving person) in the scene
'''


class UnrealCvTracking_MTMCEnv_v3(gym.Env):
    def __init__(self,
                 setting_file,
                 reset_type,
                 action_type='discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(320, 240),
                 nav='Random',  # Random, Goal, Internal
                 args=None):
        self.reset_type = reset_type
        self.action_type = action_type
        self.observation_type = observation_type
        self.reward_type = reward_type
        self.docker = docker
        self.resolution = resolution
        self.nav = nav
        self.args = args

        # load setting file
        setting = load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        # self.mask_target_list = setting['mask_targets']
        self.discrete_actions = setting['discrete_actions']
        self.discrete_target_actions = setting['discrete_target_actions']
        self.discrete_actions_player = setting['discrete_actions_player']
        self.continous_actions = setting['continous_actions']
        self.continous_actions_player = setting['continous_actions_player']
        self.max_steps = setting['max_steps']
        self.max_obstacles = setting['max_obstacles']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.objects_list = setting['objects_list']
        self.field_size = setting['field_size']
        if setting.get('reset_area'):
            self.reset_area = setting['reset_area']
        if setting.get('cam_area'):
            self.cam_area = setting['cam_area']
        if setting.get('goal_list'):
            self.goal_list = setting['goal_list']
        if setting.get('camera_loc'):
            self.camera_loc = setting['camera_loc']
        self.target_start_area = setting['safe_start']
        self.cam_start_traj = setting['cam_safe_start']
        self.target_default_z = setting['target_default_z'] if setting.get('target_default_z') else 0
        self.cam_default_z = setting['cam_default_z'] if setting.get('cam_default_z') else 200
        self.zoom_enabled = False if setting['zoom'] == 0 else True
        print("Env Name:", self.env_name)
        print('Use Zoom: ', self.zoom_enabled)
        # self.auto_move = setting['auto_move']
        if self.reset_type in [1, 2, 4]:
            self.auto_move = True
        else:
            self.auto_move = False
        if self.reset_type in [2, 3, 4, 5]:
            self.use_tracking = True
        else:
            self.use_tracking = False
        if self.reset_type in [4, 5]:
            self.use_predict = True
        else:
            self.use_predict = False
        print("Reset Type: ", self.reset_type)
        print('Target auto move: ',  self.auto_move)
        print('Use QDTrack: ', self.use_tracking)
        print("Use Predict: ", self.use_predict)
        print("Target Actions: ", len(self.discrete_target_actions))
        print("-----v3.py-----")

        # parameters for rendering map
        self.target_move = setting['target_move']
        self.camera_move = setting['camera_move']
        self.scale_rate = setting['scale_rate']
        self.pose_rate = setting['pose_rate']
        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']

        self.textures_list = misc.get_textures(setting['imgs_dir'], self.docker)
        self.num_target = len(self.target_list)
        self.num_cam = len(self.cam_id)

        # data processing
        if len(self.target_start_area) == 1:
            self.target_start_area = self.target_start_area * self.num_target
        if len(self.target_start_area[0]) == 2:
            for i in range(len(self.target_start_area)):
                self.target_start_area[i] = self.get_start_area(self.target_start_area[i], 100)

        # start unreal env and connect UnrealCV
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
        env_ip, env_port = self.unreal.start(self.docker, resolution)
        self.unrealcv = Tracking(cam_id=self.cam_id[0], port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=resolution)
        self.unrealcv.color_dict = self.unrealcv.build_color_dic(self.target_list)
        self.unrealcv.pitch = self.pitch
        self.unrealcv.set_interval(30)

        # define action and observation space (A, S)
        # assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        assert self.action_type == 'Discrete'
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray']
        self.action_space = spaces.Discrete(len(self.discrete_actions))  # only consider discrete
        self.observation_space = self.unrealcv.define_observation(self.cam_id[0], self.observation_type, 'fast')

        # target_move type
        self.unrealcv.init_objects(self.objects_list)
        if self.auto_move:
            for i, target in enumerate(self.target_list):
                self.unrealcv.auto_move(target, auto=True, time_step=5, min_speed=self.target_move[0], max_speed=self.target_move[1])
        else:
            for i, target in enumerate(self.target_list):
                self.unrealcv.auto_move(target, auto=False)

        # self.max_mask_area = np.ones(self.num_cam) * 0.001 * self.resolution[0] * self.resolution[1]
        self.count_steps = 0  # steps of tiny tracking
        self.record_eps = 0  # num of tiny tracking

        # init
        self.info = dict()
        self.states = []
        self.cam_pose = []
        self.target_pos = []
        self.cam_bbox = []
        # zoom
        self.zoom_scales = np.ones(self.num_cam)
        self.zoom_in_scale = 0.9
        self.zoom_out_scale = 1.1

        self.vis_dist = 1500
        self.unwatch_weight = 0.5

        # save to watch
        self.save_count = 0
        self.save_steps = random.sample(list(np.arange(1, self.max_steps)), k=10)
        self.save_data1 = []
        self.save_data2 = []
        self.save_data3 = []

        if self.use_tracking:
            config_file = '../mmtracking-master/configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17-private-half.py'
            checkpoint_file = '../mmtracking-master/ckpt/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17_20220315_163453-68899b0a.pth'
            self.model = init_model(config_file, checkpoint_file, device='cuda:0')  # cuda:0

        self.cut_size = (16, 16)
        self.output_camera_obs_size = (19, self.cut_size[0], self.cut_size[1])
        self.output_target_obs_size = (3 * self.num_cam + 2 * self.num_target, )

    def get_env_info(self):
        env_info = {
            "centralized_obs_shape": self.get_centralized_obs_size(),
            "n_actions": self.get_num_actions(),
            "n_agents": self.get_num_agents(),
            "target_obs_shape": self.get_target_obs_size(),
            "n_target_actions": self.get_target_num_actions(),
            "n_target_agents": self.get_target_num_agents(),
            "episode_limit": self.get_episode_len()
        }
        return env_info

    def step(self, raw_actions):
        self.info = dict(
            Done=False,
            Reward=[0 for i in range(self.num_cam)],
            Team_Reward=0,
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps,
        )
        self.current_states = self.states.copy()
        self.current_cam_pos = self.cam_pose.copy()
        self.current_target_pos = self.target_pos.copy()
        self.count_steps += 1

        # get actions on cameras
        cam_actions = []
        for i in range(self.num_cam):
            cam_actions.append(self.discrete_actions[raw_actions[0][i]])

        # get actions on target, and move the target
        if not self.auto_move:
            for i, target in enumerate(self.target_list):
                self.unrealcv.set_move(target, self.discrete_target_actions[raw_actions[1][i]][1],
                                       self.discrete_target_actions[raw_actions[1][i]][0])

        target_loc_list = []
        for i, target in enumerate(self.target_list):
            target_loc_list.append(self.unrealcv.get_obj_location(target))
        self.target_pos = target_loc_list

        # move the cameras and get new states and info for rewards
        self.cam_pose = []
        self.states, visable = [], []
        self.cam_bbox = []
        visable_rate = []  # add
        for i, cam in enumerate(self.cam_id):
            cam_loc = self.unrealcv.get_location(cam, 'hard')
            cam_rot = self.unrealcv.get_rotation(cam, 'hard')

            # target loc
            cam_pose = self.move_cam(cam_actions[i], cam_loc + cam_rot)
            self.cam_pose.append(cam_pose)
            self.unrealcv.set_location(cam, cam_pose[:3])
            self.unrealcv.set_rotation(cam, cam_pose[3:])
            raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')
            state = self.get_zoom_pic(i, raw_state)
            self.states.append(state)

            # get bbox in view
            raw_mask = self.unrealcv.read_image(cam, 'object_mask', 'fast')
            mask = self.get_zoom_pic(i, raw_mask)
            bbox_gt = self.unrealcv.get_bboxes(mask, self.target_list)  # range [0, 1]
            if not self.use_tracking:
                self.cam_bbox.append(bbox_gt)
            else:
                # get bbox from tracking
                result = inference_mot(self.model, state, frame_id=self.count_steps)
                bbox = [((0, 0), (0, 0))] * self.num_target
                for k, tracked_person in enumerate(result['track_bboxes'][0]):
                    if k < self.num_target:
                        bbox[k] = (
                            (tracked_person[1] / self.resolution[0], tracked_person[2] / self.resolution[1]),
                            (tracked_person[3] / self.resolution[0], tracked_person[4] / self.resolution[1]))
                self.cam_bbox.append(bbox)

            visable.append(self.get_target_info_via_bbox(bbox_gt))
            visable_rate.append(self.get_target_rate_via_bbox(bbox_gt))

        # get rewards
        cam_target = np.array(visable)

        rewards = cam_target.sum(axis=1) / self.num_target

        cam_target_rate = np.array(visable_rate)
        target_reward = - cam_target_rate.sum(axis=0)

        team_rewards = np.sum(cam_target.sum(axis=0) > 0) / self.num_target

        if self.count_steps >= self.max_steps:
            self.info['Done'] = True

        if self.info['Done']:
            self.record_eps += 1

        self.info['Reward'] = rewards
        self.info['Target_Reward'] = target_reward
        self.info['Team_Reward'] = team_rewards
        self.info['Cam_Pose'] = self.current_cam_pos
        self.info['Target_Pose'] = self.target_pos
        self.info['Steps'] = self.count_steps

        self.info['states'] = self.states

        return self.states, self.info['Reward'], self.info['Done'], self.info

    def reset(self):
        self.count_steps = 0
        self.states, self.cam_pose, self.target_pos = [], [], []

        self.save_steps = random.sample(list(np.arange(1, self.max_steps)), k=10)

        # reset obstacles
        self.unrealcv.clean_obstacles()
        obstacles_num = self.max_obstacles
        obstacle_scales = [[1, 1.2] if np.random.binomial(1, 0.5) == 0 else [1.5, 2] for k in range(obstacles_num)]
        self.unrealcv.random_obstacles(self.objects_list, self.textures_list,
                                       obstacles_num, self.reset_area, [0, 0, 0, 0], obstacle_scales)
        obj_loc = []
        for obj in self.objects_list:
            obj_loc.append(self.unrealcv.get_obj_location(obj))

        # reset targets
        for i, target in enumerate(self.target_list):
            target_loc = self.get_random_point_via_box(*self.target_start_area[i])
            while not self.check_safe_start(target_loc, obj_loc):
                target_loc = self.get_random_point_via_box(*self.target_start_area[i])
            # target appearance
            self.unrealcv.random_player_texture(target, self.textures_list)
            self.unrealcv.set_obj_location(target, target_loc)
            self.target_pos.append(self.unrealcv.get_obj_pose(target))
            if not self.auto_move:
                self.unrealcv.set_move(target, 0, 0)  # angle , v

        # reset cameras
        cam_start_traj = self.cam_start_traj.copy()
        random.shuffle(cam_start_traj)
        self.cam_bbox = []
        for i, cam in enumerate(self.cam_id):
            if len(cam_start_traj[i]) == 1:
                i_traj = cam_start_traj[i][0]
            elif len(cam_start_traj[i]) == 2:
                i_traj = cam_start_traj[i][0] + (cam_start_traj[i][1] - cam_start_traj[i][0]) * random.random()
            else:
                raise ValueError('len(self.cam_start_traj[i]) should be 1 or 2.')
            cam_loc = self.camera_traj2location(i_traj)
            cam_rot = [0, 360 * random.random() - 180, 0]

            self.unrealcv.set_location(cam, cam_loc)
            self.unrealcv.set_rotation(cam, cam_rot)
            self.cam_pose.append(cam_loc + cam_rot)

            raw_state = self.unrealcv.get_observation(cam, self.observation_type, 'fast')
            state = self.get_zoom_pic(i, raw_state)
            self.states.append(state)

            # add for 2d
            raw_mask = self.unrealcv.read_image(cam, 'object_mask', 'fast')
            mask = self.get_zoom_pic(i, raw_mask)
            bbox_gt = self.unrealcv.get_bboxes(mask, self.target_list)  # range [0, 1]
            if not self.use_tracking:
                self.cam_bbox.append(bbox_gt)
            else:
                # get bbox from tracking
                result = inference_mot(self.model, state, frame_id=self.count_steps)
                bbox = [((0, 0), (0, 0))] * self.num_target
                for k, tracked_person in enumerate(result['track_bboxes'][0]):
                    if k < self.num_target:
                        bbox[k] = (
                            (tracked_person[1] / self.resolution[0], tracked_person[2] / self.resolution[1]),
                            (tracked_person[3] / self.resolution[0], tracked_person[4] / self.resolution[1]))
                self.cam_bbox.append(bbox)

        self.current_states = self.states.copy()
        self.current_cam_pos = self.cam_pose.copy()
        self.current_target_pos = self.target_pos.copy()

        out = self.states.copy()
        return out

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close:
            self.unreal.close()
        imgs = np.hstack([self.states[i] for i in range(len(self.cam_id))])
        imgs = cv2.putText(imgs, "{:.2f}".format(self.info['xy_distance']), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 255, 255))
        imgs = cv2.putText(imgs, "{:.2f}".format(self.info['delta_distance']), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 255, 255))
        imgs = cv2.putText(imgs, "{:.2f}".format(self.cam_pose[0][2]), (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 255, 255))
        imgs = cv2.putText(imgs, "{:.2f}".format(self.info['Reward'][0]), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 255, 255))
        cv2.imshow("Pose-assisted-multi-camera-collaboration", imgs)
        cv2.waitKey(1)

    def to_render(self, save, save_path):
        img = map_render(self.states, self.field_size, self.cam_pose, self.target_pos)
        if save:
            cv2.imwrite(save_path, img)

    def paint_and_save(self, img, bbox_det, bbox_track, save_path):
        img = img.copy()
        for bbox in bbox_det:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        for bbox in bbox_track:
            cv2.rectangle(img, (int(bbox[1]), int(bbox[2])), (int(bbox[3]), int(bbox[4])), (0, 0, 255), 2)
            cv2.putText(img, str(int(bbox[0])), (int(bbox[1]), int(bbox[2])), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0))
        cv2.imwrite(save_path, img)

    def seed(self, seed=None):
        pass

    def get_zoom_pic(self, cam_i, picture):
        if not self.zoom_enabled:
            return picture
        zoom_picture = picture[
                       int(self.resolution[1] * (1 - self.zoom_scales[cam_i]) / 2):
                       (self.resolution[1] - int(self.resolution[1] * (1 - self.zoom_scales[cam_i]) / 2)),
                       int(self.resolution[0] * (1 - self.zoom_scales[cam_i]) / 2):
                       (self.resolution[0] - int(self.resolution[0] * (1 - self.zoom_scales[cam_i]) / 2)),
                       :]
        zoom_picture = cv2.resize(zoom_picture, self.resolution)
        return zoom_picture

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
        return start_area

    def get_random_point_via_box(self, x_min, x_max, y_min, y_max):
        x = x_min + (x_max - x_min) * random.random()
        y = y_min + (y_max - y_min) * random.random()
        z = self.target_default_z
        return [x, y, z]

    def camera_traj2location(self, traj):
        # upper right is 0, and increase anticlockwise
        traj = traj % (self.field_size[0] * 2 + self.field_size[1] * 2)
        if traj < self.field_size[0]:
            loc = [self.field_size[0]/2-traj, self.field_size[1]/2, self.cam_default_z]
        elif traj < self.field_size[0] + self.field_size[1]:
            loc = [-self.field_size[0]/2, self.field_size[1]/2-traj+self.field_size[0], self.cam_default_z]
        elif traj < self.field_size[0] * 2 + self.field_size[1]:
            loc = [-3*self.field_size[0]/2+traj-self.field_size[1], -self.field_size[1]/2, self.cam_default_z]
        else:
            loc = [self.field_size[0]/2, traj-2*self.field_size[0]-3*self.field_size[1]/2, self.cam_default_z]
        return loc

    def camera_location2traj(self, cam_loc):
        x, y, z = cam_loc
        four_lines = np.array([abs(y-self.field_size[1]/2), abs(x+self.field_size[0]/2),
                               abs(y+self.field_size[1]/2), abs(x-self.field_size[0]/2)])
        line_id = np.argmin(four_lines)
        if line_id == 0:
            traj = self.field_size[0]/2 - x
        elif line_id == 1:
            traj = self.field_size[1]/2 - y + self.field_size[0]
        elif line_id == 2:
            traj = x + 3*self.field_size[0]/2 + self.field_size[1]
        else:
            traj = y + 3*self.field_size[1]/2 + 2*self.field_size[0]
        traj = traj % (self.field_size[0] * 2 + self.field_size[1] * 2)
        return traj

    def move_cam(self, action, cam_pose):
        traj, angle = action
        old_traj = self.camera_location2traj(cam_pose[:3])
        new_cam_loc = self.camera_traj2location(old_traj + traj)
        new_cam_rot = cam_pose[3:]
        new_cam_rot[1] += angle
        return new_cam_loc + new_cam_rot

    def check_safe_start(self, loc, unsafe_list, range=100):
        unsafe_list = np.array(unsafe_list)
        unsafe_list = unsafe_list[:, :2]
        loc = np.array(loc)
        loc = loc[:2]
        near = np.sum(np.abs(unsafe_list - loc) < range, axis=1)
        if np.sum(near == 2) > 0:
            return False
        else:
            return True

    def get_target_info_via_bbox(self, bbox):
        th = 0.001
        info = []
        for i in range(len(bbox)):
            w_rate = bbox[i][1][0] - bbox[i][0][0]
            h_rate = bbox[i][1][1] - bbox[i][0][1]
            info.append(w_rate * h_rate > th)
        return info
    
    def get_target_rate_via_bbox(self, bbox):
        rate = []
        for i in range(len(bbox)):
            w_rate = bbox[i][1][0] - bbox[i][0][0]
            h_rate = bbox[i][1][1] - bbox[i][0][1]
            rate.append(w_rate * h_rate)
        return rate

    def get_centralized_obs(self):
        cam_pose_list = []
        cam_target_list = []
        save_list = []
    
        for i in range(self.num_cam):
            t_obs = np.array(self.cam_bbox[i]).reshape(-1)
            if self.use_tracking:
                t_xy = self.transfer2xy(t_obs, self.current_cam_pos[i])
                cam_target_list.append(t_xy)
            cam_info = np.array([self.current_cam_pos[i][0] / self.field_size[0],
                                 self.current_cam_pos[i][1] / self.field_size[1], self.current_cam_pos[i][4] / 180])
            save_list.append(cam_info)
            cam_info = np.tile(cam_info[:, np.newaxis, np.newaxis], (1, self.cut_size[0], self.cut_size[1]))
            cam_pose_list.append(cam_info)

        if self.use_tracking:
            targets_list = self.concat_targets_without_ID(cam_target_list)
            targets_list = self.xyz2pic_normed(targets_list)
            if self.use_predict:
                targets_list = targets_list + self.unwatched_target()
        else:
            targets_list = self.xyz2pic(self.target_pos)

        centralized_obs = []
        for i in range(self.num_cam):
            now_obs = np.concatenate([targets_list, cam_pose_list[i]], axis=0)
            for j in range(self.num_cam):
                if j != i:
                    now_obs = np.concatenate([now_obs, cam_pose_list[j]], axis=0)
            centralized_obs.append(now_obs)
    
        return centralized_obs

    def get_target_obs(self):
        cam_pose_list = []
        for i in range(self.num_cam):
            cam_info = np.array([self.current_cam_pos[i][0] / self.field_size[0],
                                 self.current_cam_pos[i][1] / self.field_size[1], self.current_cam_pos[i][4] / 180])
            cam_pose_list.append(cam_info)

        cam_obs = np.concatenate(cam_pose_list, axis=0)

        target_obs = []
        for i in range(self.num_target):
            now_obs = np.concatenate([cam_obs, self.target_pos[i][:2]], axis=0)
            for j in range(self.num_target):
                if j != i:
                    now_obs = np.concatenate([now_obs, self.target_pos[j][:2]], axis=0)
            target_obs.append(now_obs)

        return target_obs

    def computeIOU(self, boxa, boxb):
        if boxb[0] >= boxa[2] or boxb[2] <= boxa[0] or boxb[1] >= boxa[3] or boxb[3] <= boxa[1]:
            return 0
        x = np.array([boxa[0], boxa[2], boxb[0], boxb[2]])
        y = np.array([boxa[1], boxa[3], boxb[1], boxb[3]])
        x = np.sort(x)
        y = np.sort(y)
        I = (x[2] - x[1]) * (y[2] - y[1])
        U = (boxa[2] - boxa[0]) * (boxa[3] - boxa[1]) + (boxb[2] - boxb[0]) * (boxb[3] - boxb[1]) - I
        return I / U

    def get_centralized_obs_size(self):
        # return (19, 16, 16)
        return self.output_camera_obs_size

    def get_target_obs_size(self):
        return self.output_target_obs_size

    def xyz2pic(self, pose_list):
        x_size, y_size = self.cut_size[0], self.cut_size[1]
        pic = np.zeros((1, x_size, y_size))
        for target_pose in pose_list:
            xid = int(target_pose[0] * x_size / self.field_size[0] + x_size / 2)
            yid = int(target_pose[1] * x_size / self.field_size[1] + y_size / 2)
            pic[0, xid, yid] += 1
        return pic

    def xyz2pic_normed(self, pose_list):
        x_size, y_size = self.cut_size[0], self.cut_size[1]
        pic = np.zeros((1, x_size, y_size))
        for target_pose in pose_list:
            if target_pose[0] == -1:
                continue
            xid = np.clip(int(target_pose[0] * x_size), 0, self.cut_size[0]-1)
            yid = np.clip(int(target_pose[1] * x_size), 0, self.cut_size[1]-1)
            pic[0, xid, yid] += 1
        return pic

    def unwatched_target(self):
        x_size, y_size = self.cut_size[0], self.cut_size[1]
        watch_map = np.ones((1, x_size, y_size)) * self.unwatch_weight
        for k in range(self.num_cam):
            for i in range(x_size):
                for j in range(y_size):
                    cam_info = np.array([self.current_cam_pos[k][0], self.current_cam_pos[k][1] / self.field_size[1],
                                         self.current_cam_pos[k][4]])
                    if self.is_watched(cam_info, [-self.field_size[0] / 2 + self.field_size[0] / x_size * (-0.5 + i),
                                                  -self.field_size[1] / 2 + self.field_size[1] / y_size * (-0.5 + j)],
                                       self.vis_dist):
                        watch_map[0][i][j] = 0
        return watch_map

    def is_watched(self, cam_pose, target_pos, vdis):
        """
        input data is not normed
        :param cam_pose: [x, y, theta]   theta -180~180
        :param target_pos: [x, y]
        :param vdis: int
        :return: True or False
        """
        x_dis = target_pos[0] - cam_pose[0]
        y_dis = target_pos[1] - cam_pose[1]

        if x_dis * x_dis + y_dis * y_dis > vdis * vdis:
            return False

        if x_dis > 0:
            theta = np.arctan(y_dis / x_dis) / np.pi * 180
        elif x_dis < 0 and y_dis >= 0:
            theta = np.arctan(y_dis / x_dis) / np.pi * 180 + 180
        elif x_dis < 0 and y_dis < 0:
            theta = np.arctan(y_dis / x_dis) / np.pi * 180 - 180
        else:
            theta = 90 * np.sign(y_dis)

        if np.abs(theta - cam_pose[2]) < 45 or np.abs(theta - cam_pose[2]) > 315:
            return True
        else:
            return False

    def predict_info(self, last_obs, now_obs):
        out = 2 * now_obs - last_obs
        for i in range(self.num_target):
            if not np.any(last_obs[4 * i:4 * (i + 1)]):
                out[4 * i:4 * (i + 1)] = now_obs[4 * i:4 * (i + 1)]
            if not np.any(now_obs[4 * i:4 * (i + 1)]):
                out[4 * i:4 * (i + 1)] = last_obs[4 * i:4 * (i + 1)]
        return out

    def predict_info2(self, last_obs, now_obs):
        out = 2 * now_obs - last_obs
        out2 = 3 * now_obs - 2 * last_obs
        for i in range(self.num_target):
            if not np.any(last_obs[4 * i:4 * (i + 1)]):
                out[4 * i:4 * (i + 1)] = now_obs[4 * i:4 * (i + 1)]
                out2[4 * i:4 * (i + 1)] = now_obs[4 * i:4 * (i + 1)]
            if not np.any(now_obs[4 * i:4 * (i + 1)]):
                out[4 * i:4 * (i + 1)] = last_obs[4 * i:4 * (i + 1)]
                out2[4 * i:4 * (i + 1)] = last_obs[4 * i:4 * (i + 1)]
        return out, out2

    def predict_via_xyinfo(self, last_obs, now_obs):
        out = 2 * now_obs - last_obs
        last_vision = last_obs > -1
        now_vision = now_obs > -1
        for i in range(int(len(now_obs) / 2)):
            if not np.any(last_vision[2*i:2*(i+1)]):
                out[2*i:2*(i+1)] = now_obs[2*i:2*(i+1)]
            if not np.any(now_vision[2*i:2*(i+1)]):
                out[2 * i:2 * (i + 1)] = last_obs[2 * i:2 * (i + 1)]
        return out

    def transfer2xy(self, bbox_list, cam_info):
        # norm to 0-1
        xy_list = np.array([])
        for i in range(self.num_target):
            if np.any(bbox_list[4*i:4*(i+1)]) and bbox_list[4*i+2] - bbox_list[4*i] > 0.01 and bbox_list[4*i+3] - bbox_list[4*i+1] > 0.01:
                cam_pose = [cam_info[0], cam_info[1], cam_info[4] / 180 * np.pi]
                uv_coor = [(bbox_list[4*i] + bbox_list[4*i+2]) / 2 * 2 - 1, bbox_list[4*i+3] * 2 - 1]
                out = self.uv2xyz(uv_coor, cam_pose)
                norm_out = np.array([out[0][0] / self.field_size[0] + 0.5, out[1][0] / self.field_size[1] + 0.5])
            else:
                norm_out = np.array([-1., -1.])
            xy_list = np.append(xy_list, norm_out)
        return xy_list

    def transfer2xy_gt(self, bbox_list):
        # norm to 0-1
        xy_list = np.array([])
        for i in range(self.num_target):
            if np.any(bbox_list[4 * i:4 * (i + 1)]):
                norm_out = np.array([self.target_pos[i][0] / self.field_size[0] + 0.5, self.target_pos[i][1] / self.field_size[1] + 0.5])
            else:
                norm_out = np.array([-1., -1.])
            xy_list = np.append(xy_list, norm_out)
        return xy_list

    def concat_targets(self, loc_list):
        loc_list = np.array(loc_list)
        index = loc_list > -1
        out = np.array([])
        for i in range(loc_list[0].size):
            tmp_loc_list = loc_list[:, i]
            tmp_index = index[:, i]
            if np.sum(tmp_index) == 0:
                out = np.append(out, -1)
            else:
                out = np.append(out, np.mean(tmp_loc_list[tmp_index]))
        return out

    def concat_targets_without_ID(self, loc_list):
        count_list = []
        for i in range(self.num_target):
            if loc_list[0][2 * i] > -1:
                count_list.append(1)
            else:
                break
        out = loc_list[0].copy()
        out = np.append(out, np.array([-1] * 2 * self.num_target))
        for i in range(1, len(loc_list)):
            out, count_list = self.concat_two_targets_list(out, loc_list[i], count_list)
        # attention
        out = np.array(sorted(out.reshape(-1, 2), key=lambda x: (x[0], x[1]), reverse=True))
        return out

    def concat_two_targets_list(self, target_list_a, target_list_b, count_list):
        th = 0.2
        list_a = target_list_a.reshape(-1, 2)
        list_b = target_list_b.reshape(-1, 2)
        out = list_a.copy()

        for i in range(list_b.shape[0]):
            if list_b[i][0] > -1:
                dis = np.sum(np.abs(list_a - list_b[i]), axis=1)
                min_id = np.argmin(dis)
                if dis[min_id] < th:
                    out[min_id] = (out[min_id] * count_list[min_id] + list_b[i]) / (count_list[min_id] + 1)
                    count_list[min_id] += 1
                elif len(count_list) < 2 * self.num_target:
                    out[len(count_list)] = list_b[i]
                    count_list.append(1)
                else:
                    print("Targets are larger than 20.")
            else:
                break

        return out.reshape(-1), count_list

    def uv2xyz(self, t4p_norm, cam_pose):
        t4p = [t4p_norm[0] * self.resolution[0], t4p_norm[1] * self.resolution[1]]
        fcous = 300
        Xc = 200 * fcous / t4p[1]

        t4c = np.mat([[Xc], [Xc * t4p[0] / fcous], [-200]])
        R = np.mat([[np.cos(cam_pose[2]), np.sin(cam_pose[2]), 0],
                    [-np.sin(cam_pose[2]), np.cos(cam_pose[2]), 0],
                    [0, 0, 1]])
        c4g = np.mat([[cam_pose[0]], [cam_pose[1]], [200]])
        t4g = np.matmul(np.linalg.inv(R), t4c) + c4g
        return t4g

    def get_num_actions(self):
        return len(self.discrete_actions)

    def get_num_agents(self):
        return self.num_cam

    def get_target_num_actions(self):
        return len(self.discrete_target_actions)

    def get_target_num_agents(self):
        return self.num_target

    def get_episode_len(self):
        return self.max_steps

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list.
        If all actions are legal, return a full 1 matrix of size [n_agents, n_actions]"""
        return np.ones((self.get_num_agents(), self.get_num_actions()))

    def get_target_avail_actions(self):
        return np.ones((self.get_target_num_agents(), self.get_target_num_actions()))

    def get_verti_direction(self, current_pose, target_pose):
        person_height = target_pose[2]
        plane_distance = self.unrealcv.get_distance(current_pose, target_pose, 2)
        height = current_pose[2] - person_height
        angle = np.arctan2(height, plane_distance) / np.pi * 180
        angle_now = angle + current_pose[-1]
        return angle_now

    def adjust_rotation(self, cam_pose, target_pose):
        x_delt = target_pose[0] - cam_pose[0]
        y_delt = target_pose[1] - cam_pose[1]
        height = cam_pose[2] - target_pose[2]
        plane_distance = math.sqrt(x_delt * x_delt + y_delt * y_delt)
        pitch = - np.arctan2(height, plane_distance) / np.pi * 180
        yaw = np.arctan2(y_delt, x_delt) / np.pi * 180
        roll = 0
        return [roll, yaw, pitch]

    def nearby_goal(self, pose, diff):
        x1 = max(self.reset_area[0], pose[0] - diff)
        x2 = min(self.reset_area[1], pose[0] + diff)
        y1 = max(self.reset_area[2], pose[1] - diff)
        y2 = min(self.reset_area[3], pose[1] + diff)
        return [x1, x2, y1, y2]


def map_render(img_list, field_size=None, cam_pose=None, target_pose=None):
    row = 2
    column = round(len(img_list) / 2)
    img_width = img_list[0].shape[1]
    img_height = img_list[0].shape[0]
    img = np.zeros((img_height * row, img_width * column + 500, 3)) + 255

    # camera vision
    for i in range(row):
        for j in range(column):
            img[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width, :] = img_list[
                i * column + j]

    # env
    scale = min(400 / field_size[0], (img_height * row - 100) / field_size[1])
    offset_x = img_width * column + 250
    offset_y = img_height * row / 2
    cv2.rectangle(img, (int(offset_x - field_size[0] / 2 * scale), int(offset_y - field_size[1] / 2 * scale)),
                  (int(offset_x + field_size[0] / 2 * scale), int(offset_y + field_size[1] / 2 * scale)),
                  color=(0, 0, 0), thickness=2)
    for i, cam in enumerate(cam_pose):
        cv2.circle(img, (int(offset_x + cam[0] * scale), int(offset_y + cam[1] * scale)), radius=5,
                   color=(0, 0, 255),
                   thickness=-1)  # BGR
        cv2.putText(img, str(i + 1), (int(offset_x + cam[0] * scale), int(offset_y + cam[1] * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0))
        # direction
        # cv2.line(img, (int(offset_x + cam[0] * scale), int(offset_y + cam[1] * scale)),
        #          (int(offset_x + cam[0] * scale + 50 * np.cos(cam[4] / 180 * np.pi)),
        #           int(offset_y + cam[1] * scale + 50 * np.sin(cam[4] / 180 * np.pi))),
        #          color=(0, 0, 0), thickness=1)
        # range
        cv2.line(img, (int(offset_x + cam[0] * scale), int(offset_y + cam[1] * scale)),
                 (int(offset_x + cam[0] * scale + 50 * np.cos((cam[4] - 45) / 180 * np.pi)),
                  int(offset_y + cam[1] * scale + 50 * np.sin((cam[4] - 45) / 180 * np.pi))),
                 color=(0, 0, 0), thickness=1)
        cv2.line(img, (int(offset_x + cam[0] * scale), int(offset_y + cam[1] * scale)),
                 (int(offset_x + cam[0] * scale + 50 * np.cos((cam[4] + 45) / 180 * np.pi)),
                  int(offset_y + cam[1] * scale + 50 * np.sin((cam[4] + 45) / 180 * np.pi))),
                 color=(0, 0, 0), thickness=1)
    for i, target in enumerate(target_pose):
        cv2.circle(img, (int(offset_x + target[0] * scale), int(offset_y + target[1] * scale)), radius=3,
                   color=(255, 0, 0), thickness=-1)

    return img
