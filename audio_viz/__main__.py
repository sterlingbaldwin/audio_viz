import sys
import yaml
import argparse
from time import sleep
from pathlib import Path

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from pyartnet import ArtNetNode

def load_yaml(filepath):
    with open(filepath, 'r') as instream:
        return yaml.load(instream, Loader=yaml.SafeLoader)


def main():
    parser = argparse(desc="A tool for controlling stage lights over DMX")
    parser.add_argument(
        '-c', '--config', 
        default="../shows/whale/config.yaml",
        help="Path to yaml config file")
    parser.add_argument(
        '-s', '--script', 
        default="../shows/whale/script.yaml",
        help="Path to yaml lighting script file")
    parser.add_argument(
        '-v', '--visualize', 
        action="store_true",
        help="Turn on the wave form visualization")
    args = parser.parse_args()
    
    if not (config_path := Path(args['config'])).exists():
        print(f"ERROR: No config file found at {config_path}")
        return 1
    else:
        config = load_yaml(config_path)

    if not (script_path := Path(args['script'])).exists():
        print(f"ERROR: No script file found at {script_path}")
        return 1
    else:
        script = load_yaml(script_path)
    

    # sampling information
    Fs = config.get('audio_sample_rate', 44100)   # sample rate
    loop_delay = config.get('loop_delay', 0.0001)   # sample rate
    T = 1 / Fs  # sampling period
    t = 0.1     # seconds of sampling
    N = Fs * t  # total points in signal
    freq = 1.0 / config.get('hrz', 30)

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    stream = p.open(
        format=sample_format,
        channels=1,
        rate=fs,
        input=True)
    
    is args.visualization == True:
        fig, ax = plt.subplots()
    
    
    while True:
        
        
        is args.visualization == True:
            plt.pause(loop_delay)
        else:
            sleep(loop_delay)

    

    return 0

if __name__ == "__main__":
    sys.exit(main())