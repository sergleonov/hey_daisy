import argparse
import queue
import sys
import sounddevice as sd

from vosk import Model, KaldiRecognizer
import json
import time

import logging
import math
import signal
import threading
from sys import platform


import cv2
import math
import numpy as np
from PIL import Image

import bosdyn.client
import bosdyn.client.util
from bosdyn import geometry
from bosdyn.api import geometry_pb2, image_pb2, trajectory_pb2, world_object_pb2
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b,
                                        get_vision_tform_body, get_odom_tform_body)
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, blocking_sit, blocking_command
from bosdyn.client.robot_id import RobotIdClient, version_tuple
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
from scipy.spatial.transform import Rotation as R


BODY_LENGTH = 1.1

class CommandFollower(object):

    def __init__(self, robot):

        # instantiate robot
        self._robot = robot
        self._robot_id = robot.ensure_client(RobotIdClient.default_service_name).get_id(timeout=0.4)
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)

        # Stopping Distance (x,y) offset from the tag and angle offset from desired angle.
        self._tag_offset = 0.5 + BODY_LENGTH / 2.0  # meters

        # Maximum speeds.
        self._max_x_vel = 0.5
        self._max_y_vel = 0.5
        self._max_ang_vel = 1.0
        self._rotation_duration = 5

        # self._avoid_obstacles = options.avoid_obstacles  # Disable obstacle avoidance.
        self._avoid_obstacles = True

        # Counter for the number of iterations completed.
        self._iterations = 0

        # Indicator for if motor power is on.
        self._powered_on = False

        # Maximum amount of iterations before powering off the motors.
        self._max_iterations = 100000

        # Camera intrinsics for the current camera source being analyzed.
        self._intrinsics = None
        # Transform from the robot's camera frame to the baselink frame.
        # It is a math_helpers.SE3Pose.
        self._camera_tform_body = None
        # Transform from the robot's baselink to the world frame.
        # It is a math_helpers.SE3Pose.
        self._body_tform_world = None

        # List of all possible camera sources.
        self._source_names = [
            src.name for src in self._image_client.list_image_sources() if
            (src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and 'depth' not in src.name)
        ]

        # Camera source which a bounding box was last detected in.
        self._previous_source = None

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_client.get_robot_state()
    
    def get_desired_angle(self, xhat):
        """Compute heading based on the vector from robot to object."""
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.array([xhat, yhat, zhat]).transpose()
        return Quat.from_matrix(mat)
    
    def do_stop(self):
        stop_cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.0, v_y=0.0, v_rot=0.0)
        self._robot_command_client.robot_command(lease=None, command=stop_cmd)
        blocking_stand(self._robot_command_client)
        time.sleep(0.5)
        return
    
    def do_stand(self):
        blocking_stand(self._robot_command_client, timeout_sec=5)
        time.sleep(0.5)
        return
    
    def do_sit(self):
        blocking_sit(self._robot_command_client, timeout_sec=5)
        time.sleep(0.5)
        return
    
    def rotate(self, angle: float, direction: int):
        if direction != 1 and direction != -1:
            print("Invalid direction parameter. Rotation failed")
            return
        
        frame_snapshot = self.robot_state.kinematic_state.transforms_snapshot
        v_x, v_y, v_rot = 0.0, 0.0, -angle/self._rotation_duration if direction < 0 else angle/self._rotation_duration

        rotation_cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=v_x, v_y=v_y, v_rot=v_rot,
            frame_name="body"
        )
        blocking_command(
            self._robot_command_client, 
            rotation_cmd, 
            check_status_fn=lambda status: True,
            end_time_secs=time.time() + self._rotation_duration,
            timeout_sec=self._rotation_duration + 1.0,
            update_frequency=10.0
        )
        time.sleep(self._rotation_duration)

    def power_on(self):
        """Power on the robot."""
        self._robot.power_on()
        self._powered_on = True
        print(f'Powered On {self._robot.is_powered_on()}')

    def power_off(self):
            """Power off the robot."""
            self._robot.power_off()
            print(f'Powered Off {not self._robot.is_powered_on()}')
    
    def start(self):
        """Claim lease of robot and start the fiducial follower."""
        self._robot.time_sync.wait_for_sync()
        # Stand the robot up.
        self.power_on()
        blocking_stand(self._robot_command_client)
        time.sleep(0.35)


# global queue for waveform input
q = queue.Queue()   

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def degrees_to_radians(angle: int) -> float:
    return angle/180.0 * math.pi


# method for reading the sound data from microphone and converting it into a string
def get_text_from_audio(model, device) -> str:
    # Args: Vosk STT model, audio input device
    # Returns: Transcribed text string, or timeout message if no speech
    # Purpose: Captures audio via microphone for a few seconds and transcribes to text using Vosk model

    # Gets default samplerate from device info
    device_info = sd.query_devices(device, "input")
    samplerate = int(device_info["default_samplerate"])

    # start listening for voice 
    with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=device,
                dtype="int16", channels=1, callback=callback):

        recording = KaldiRecognizer(model, samplerate)

        timeout = 5 #seconds
        start = time.time()
        text = ""
        print("Recording audio now...")
        while time.time() < start + timeout:
            data = q.get()
            if recording.AcceptWaveform(data):
                text += json.loads(recording.Result())["text"] + " "
        if text != "":
            return text
        else: 
            return "Timed out... No command given."  

# function for parsing text that model returned into command code
def parse_text_into_command(text: str, parse_list: dict) -> int:
    # Args: Transcribed text string, dictionary of keywords for a given language
    # Example parse_list: 
    # {
    #    "sit": ["sit"],
    #     "stand": ["stand"]
    # }
    # Returns: Integer command code (0:stop, 1:sit, 2:stand, -1:none)
    # Purpose: Identifies command from text by checking against keywords from a JSON file
    # Uses sets for efficient keyword intersection checks
    
    # find the command code from text
    for command, keywords in parse_list.items():
        for keyword in keywords:
            if keyword in text:
                return command
    return "none"

# function for verifying the command that was parsed
def verify_command(cmd: str, dispatch: dict) -> bool:
    # Args: Integer command code
    # Returns: Boolean (True: execution approved, False: not approved)
    # Purpose: Prompts user for approval of the parsed command, returns their decision
    # Loops until 'y' or 'n' is entered, defaults to False on empty input

    if cmd.lower() not in [*dispatch]:
        print("Command was not recognized or was invalid...")
        return False
    
    user_input = "empty"
    print(f"Received instruction to {cmd.lower()}. Do you approve execution? (y/[N])")

    while user_input != "y" and user_input != "n":
        user_input = input().strip().lower()
        if user_input == "": 
            return False
    return user_input == 'y'

def main():
    # parse arguments 
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-l", "--list-devices", action="store_true",
        help="show list of audio devices and exit")
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        "-d", "--device", type=int_or_str,
        help="input device (numeric ID or substring)")
    parser.add_argument(
        "-m", "--model", type=str, help="language model; e.g. en-us (English), ru (Russian), es (Spanish); default is en-us")
    args = parser.parse_args(remaining)


    if args.model is None:
        lang = "en-us"
        model = Model(lang)

    else:
        model = Model(lang=args.model)
        lang = args.model

    # Load a json file containing the lists of keywords to look for per command
    # Example json file:
    # {
    #     "en-us":{
    #         "sit": ["sit"]
    #         "stand": ["stand"]
    #     }
    # }

    f = open("parse_lists.json", "r")
    keyword_list = json.loads(f.read())
    f.close()

    # initialize Daisy
    sdk = bosdyn.client.create_standard_sdk("voice_listener")
    robot = sdk.create_robot("192.168.80.3")
    bosdyn.client.util.authenticate(robot)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    command_follower = CommandFollower(robot)

    # map commands to method calls
    dispatch = {
        "stop": command_follower.do_stop,
        "sit": command_follower.do_sit,
        "stand": command_follower.do_stand,
        "rotate clockwise": lambda: command_follower.rotate(degrees_to_radians(90), 1),
        "rotate counterclockwise": lambda: command_follower.rotate(degrees_to_radians(90), -1),
        "turn around": lambda: command_follower.rotate(degrees_to_radians(180), 1)
    }

    iterations = 0
    max_iterations = 50
    time.sleep(.1)  
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command_follower.start()
        while iterations < max_iterations:
            text = get_text_from_audio(model, args.device)
            print(f"Daisy heard: \"{text}\"")
            cmd = parse_text_into_command(text, keyword_list[lang])
            if verify_command(cmd, dispatch):
                print(f"Executing {cmd.lower()} command...")
                dispatch[cmd]()
            else:
                print("Command not approved")
            iterations += 1


if __name__=="__main__":
    main()