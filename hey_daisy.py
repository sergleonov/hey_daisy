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
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, blocking_sit
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
        return
    
    def do_stand(self):
        blocking_stand(self._robot_command_client, timeout_sec=5)
        return
    
    def do_sit(self):
        blocking_sit(self._robot_command_client, timeout_sec=5)
        return
    

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



q = queue.Queue()   

command_list = ["Stop", "Sit", "Stand", "None"]

commands_to_codes = {
    "stop_cmd": 0,
    "sit_cmd": 1,
    "stand_cmd": 2,
    "none": -1
}

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
def parse_text_into_command(text: str, lang: str) -> int:
    # Args: Transcribed text string, language code ("en-us", "ru")
    # Returns: Integer command code (0:stop, 1:sit, 2:stand, -1:none)
    # Purpose: Identifies command from text by checking against keywords from a JSON file
    # Uses sets for efficient keyword intersection checks

    # set of words transcribed for quick access
    words_heard = set(text.split())
    
    # load a json file containing the lists of keywords to look for per command
    f = open("parse_lists.json", "r")
    parse_list = json.loads(f.read())[lang]
    f.close()

    # get the sets of keywords for quick intersection checks (indexed by codes)
    command_sets = [
        set(parse_list["stop"]),
        set(parse_list["sit"]),
        set(parse_list["stand"])
    ]

    # check whether any command is recognized, return its code
    for i in range(len(command_sets)):
        if not command_sets[i].isdisjoint(words_heard):
            return i
    return  commands_to_codes["none"]

# function for verifying the command that was parsed
def verify_command(code: int) -> bool:
    # Args: Integer command code
    # Returns: Boolean (True: execution approved, False: not approved)
    # Purpose: Prompts user for approval of the parsed command, returns their decision
    # Loops until 'y' or 'n' is entered, defaults to False on empty input

    if code < 0 or code >= len(command_list) - 1:
        print("Invalid command code...")
        return False

    if code == commands_to_codes["none"]:
        print("Command was not recognized...")
        return False

    user_input = "empty"
    print(f"Received instruction to {command_list[code].lower()}. Do you approve execution? (y/[N])")

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

    # initialize Daisy
    sdk = bosdyn.client.create_standard_sdk("voice_listener")
    robot = sdk.create_robot("192.168.80.3")
    bosdyn.client.util.authenticate(robot)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    command_follower = CommandFollower(robot)
    iterations = 0
    max_iterations = 50
    time.sleep(.1)  
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command_follower.start()
        while iterations < max_iterations:
            text = get_text_from_audio(model, args.device)
            print(f"Daisy heard: \"{text}\"")
            code = parse_text_into_command(text, lang)
            if verify_command(code):
                match code:
                    case 0:
                        print(f"Executing {command_list[code].lower()} command")
                        command_follower.do_stop()
                    case 1:
                        print(f"Executing {command_list[code].lower()} command")
                        command_follower.do_sit()
                    case 2:
                        print(f"Executing {command_list[code].lower()} command")
                        command_follower.do_stand()
            else:
                print("Command not approved")
            iterations += 1


if __name__=="__main__":
    main()