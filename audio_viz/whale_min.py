
from random import randint
from dataclasses import dataclass
from telnetlib import Telnet
from time import sleep
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import argparse
import yaml
import sys

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pydub import AudioSegment


from pprint import pprint

import threading
import queue

OFF = [0,0,0,0]
ON = [100, 100, 100, 100]


def add_input(input_queue):
    while True:
        input_queue.put(sys.stdin.read(1))

@dataclass
class Fixture:
    channel: int    # the channel id for the light
    width: int = 4     # the light width, i.e. RGB on sepperate channels
    x: int     = 0    # the x position of the light
    y: int     = 0    # the y position of the light
    val: list[int] = field(default_factory=lambda: [100,100,100,100])
    has_updated = False
    
    def __str__(self):
        return f"channel: {self.channel}, width: {self.width}, val: {self.val}"


@dataclass
class Fade:
    fixture: Fixture     # the fixture to be faded
    current: list[int]  # the current/start RGB value of the fade
    target: list[int]   # the target RGB value of where to fade to
    step: int            # the step size
    initial: list[int]  # the starting value of the fade

    def __str__(self):
        return f"Fixture: {self.fixture}, current: {self.current}, target: {self.target}, step: {self.step}"

def scale(old_max, old_min, new_max, new_min, old_value):
    return ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

def print_lights(lights, shape):
    cells = [[0 for _ in range(shape[1])] for _ in range(shape[0])]    
    for i in range(shape[0]):
        for j in range(shape[1]):
            light = lights['house'][i][j]
            if light is not None:
                cells[i][j] = light.val

    print(
        '\n'.join(
            [''.join([f'|{item[0]:2}.{item[1]:2}.{item[2]:2}.{item[3]:2}|' if item else '|           |' for item in row]) 
                for row in cells]), flush=True)
    print('\n')


lights = {
    'pieces' : [
        Fixture(999, 0, 0, 0),
        Fixture(998, 0, 0, 0),
    ],
    'house': [
        [
            None,                   
            None,                  
            None,                  
            Fixture(573, 4, 3, 0), 
            Fixture(577, 4, 4, 0), 
            Fixture(581, 4, 5, 0), 
            Fixture(585, 4, 6, 0), 
            Fixture(589, 4, 7, 0), 
            None,                  
            None                   
        ], [
            None,                  
            None,                  
            None,                  
            Fixture(553, 4, 3, 1), 
            Fixture(557, 4, 4, 1), 
            Fixture(561, 4, 5, 1), 
            Fixture(565, 4, 6, 1), 
            Fixture(569, 4, 7, 1), 
            None,                  
            None                   
        ], [               
            Fixture(521, 4, 1, 2), 
            Fixture(525, 4, 2, 2), 
            Fixture(529, 4, 2, 2), 
            Fixture(533, 4, 3, 2), 
            Fixture(537, 4, 4, 2), 
            Fixture(541, 4, 5, 2), 
            Fixture(545, 4, 6, 2), 
            Fixture(549, 4, 7, 2), 
            None,                  
            None                   
        ], [
            Fixture(761, 4, 0, 3), 
            Fixture(757, 4, 1, 3), 
            None,                  
            None,                  
            None,                  
            None,                  
            None,                  
            None,                 
            Fixture(593, 4, 8, 3),
            Fixture(617, 4, 9, 3),
        ], [ 
            Fixture(733, 4, 0, 4),
            None,
            Fixture(753, 4, 2, 4),
            None,
            None,
            None,
            None,
            None,
            Fixture(597, 4, 8, 4),
            Fixture(621, 4, 9, 4),
        ], [ 
            Fixture(729, 4, 0, 5),
            None,
            Fixture(749, 4, 2, 5),
            None,
            None,
            None,
            None,
            None,
            None,
            Fixture(625, 4, 9, 5),
        ], [ 
            Fixture(725, 4, 0, 6),
            None,
            Fixture(745, 4, 2, 6),
            None,
            None,
            None,
            None,
            None,
            Fixture(601, 4, 8, 6),
            None
        ], [ 
            Fixture(721, 4, 0, 7),
            None,
            Fixture(741, 4, 2, 7),
            None,
            None,
            None,
            None,
            None,
            Fixture(605, 4, 8, 7),
            None
        ], [ 
            Fixture(717, 4, 0, 8),
            None,
            Fixture(737, 4, 2, 8),
            None,
            None,
            None,
            None,
            None,
            Fixture(609, 4, 8, 8),
            Fixture(629, 4, 9, 8),
        ], [ 
            Fixture(713, 4, 0, 9),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Fixture(613, 4, 8, 9),
            Fixture(633, 4, 9, 9),
        ],[ 
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Fixture(641, 4, 8, 9),
            Fixture(637, 4, 9, 9),
        ],[ 
            Fixture(673, 4, 8, 9),
            Fixture(669, 4, 8, 9),
            Fixture(665, 4, 8, 9),
            Fixture(661, 4, 8, 9),
            Fixture(657, 4, 8, 9),
            Fixture(653, 4, 8, 9),
            Fixture(649, 4, 8, 9),
            Fixture(645, 4, 8, 9),
            None,
            None,
        ],[             
            None,
            None,
            Fixture(713, 4, 8, 9),
            Fixture(709, 4, 8, 9),
            Fixture(705, 4, 8, 9),
            Fixture(701, 4, 8, 9),
            Fixture(697, 4, 8, 9),
            Fixture(693, 4, 8, 9),
            Fixture(689, 4, 8, 9),
            Fixture(685, 4, 8, 9)
        ]
    ]
}


class Whale(object):
    def __init__(self, host, port, user, password, sim=False):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.sim = sim
        self.con = None
        self.total_sent = 0
        self.message_queue = [] # list of dicts with "channel": "value"
    
    def init_connection(self):
        if self.con is not None:
            self.con.close()
            del self.con
        self.total_sent = 0
        self.con = Telnet()
        self.con.open(self.host, port=self.port)
        self.con.write(f"login {self.user} {self.password}\r\n".encode('ascii'))

    
    def flush(self):
        # print()
        message = ""
        for item in self.message_queue:
            channel, value = item.items()
            message += f"dmx {channel[1]} at {value[1]}\r\n "
        if self.sim:
            self.message_queue = []
            return
        # print(message.encode('ascii'))
        if message and self.con.get_socket():
            # print(message.encode('ascii'))
            self.total_sent += len(message)
            if self.total_sent >= 65000:
                self.init_connection()
                self.total_sent = 0
            self.con.write(message.encode('ascii'))
        self.message_queue = []


    def __enter__(self):
        if self.sim:
            print("Running in simulation mode")
            return self
        self.init_connection()
        # self.con.open(self.host, port=self.port)
        # print(self.con.read_until(b"login !").decode('ascii'))
        # self.con.write(f"login {self.user} {self.password}\r\n".encode('ascii'))
        print("Whale session initialized")
        return self

    def __exit__(self, type, value, traceback):
        print("RELEASE")
        self.con.write("Off DMX thru\r\n".encode('ascii'))
        sleep(0.1)
        self.con.close()

def set_fixture(whale: Whale, fixture: Fixture, target: list[int]):
    for idx, channel in enumerate(range(fixture.channel, fixture.channel + fixture.width)):
        whale.message_queue.append({
            'channel': channel,
            'value': target[idx]
        })

def update_fades(whale:Whale, fades:list[Fade], cycle=True):
    to_remove = []
    global lights

    for fade in fades:
        new_val = fade.current[:]
        for idx, item in enumerate(fade.current):
            # print(f"item[idx]: {item}[{idx}], target: {fade.target[idx]}, step: {fade.step}")
            if item == 0 and item == fade.target[idx] and fade.current != fade.target:
                continue

            if fade.step > 0:# and item < fade.target[idx]:
                new_val[idx] = item + fade.step
                # print(f"new_val: {new_val[idx]}")
                if new_val[idx] >= fade.target[idx]:
                    new_val[idx] = fade.target[idx]

            elif fade.step < 0:
                new_val[idx] = item + fade.step
                if new_val[idx] < 0:
                    new_val[idx] = 0
                
            if fade.current[idx] != new_val[idx]:
                # print(f"[{fade.fixture.x}][{fade.fixture.y}] old val: {fade.current} new val: {new_val} step: {fade.step}")
                # print(f"updating fixture value from {fade.current[idx]} to {new_val[idx]}")
                fade.current[idx] = new_val[idx]
                fade.fixture.val[idx] = new_val[idx]
                fade.fixture.has_updated = True
                # print(f"after update {fade.current[idx]}")
                whale.message_queue.append({
                    'channel': fade.fixture.channel + idx,
                    'value': fade.current[idx]
                })

            # if new_step != fade.step and idx == len(fade.current) -1:
            #     fade.step = new_step
            if idx == len(fade.current)-1 and fade.current >= fade.target and cycle:
                # fade.step = -1 * fade.step
                fade.step = -100
                # fade.step = -30

        # if fade.current == [0, 0, 0, 0]:
        all_zero = True
        for val in fade.current:
            if val != 0:
                all_zero = False
                break
        if all_zero:
            to_remove.append(fade)

    for fade in to_remove:
        # print(f'removing {fade}')
        fades.remove(fade)

    # print(len(fades))
    return fades


def wave(whale:Whale, light_grid:list, shape:tuple, color:list[int], fade_rate:int = 5, wave_speed:float = 0.01, source:tuple = (0, 0), small=False):
    print(wave_speed)
    cells = [[False for _ in range(shape[1])] for _ in range(shape[0])]    
    fades = []
    fade_step = max(color)//fade_rate

    cell_count = 0
    small_max = 9

    cells[source[0]][source[1]] = True
    source_fixture = light_grid[source[0]][source[1]]
    if source_fixture is not None:
        fade = Fade(
            source_fixture, 
            source_fixture.val, 
            color[:], 
            fade_step,
            source_fixture.val[:])
        fades.append(fade)

    changed = True
    while changed:
        changed = False
        to_update = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                if cells[i][j] == False:
                    continue
                if i != False and cells[i-1][j] == 0:
                    to_update.append({
                        'val': cells[i-1][j],
                        'x': i-1,
                        'y': j
                    })
                if i != shape[0]-1 and cells[i+1][j] == False:
                    to_update.append({
                        'val': cells[i+1][j],
                        'x': i+1,
                        'y': j
                    })
                if j != 0 and cells[i][j-1] == False:
                    to_update.append({
                        'val': cells[i][j-1],
                        'x': i,
                        'y': j-1
                    })
                if j != shape[1]-1 and cells[i][j+1] == False:
                    to_update.append({
                        'val': cells[i][j+1],
                        'x': i,
                        'y': j+1
                    })
        for cell in to_update:
            cell_count += 1
            if small and cell_count > small_max:
                break
            cells[ cell['x'] ][ cell['y'] ] = True
            fix = light_grid[ cell['x'] ][ cell['y'] ]
            if fix is not None:
                found = False
                for f in fades:
                    if f.fixture == fix:
                        found = True
                        break
                if not found:
                    fade = Fade(fix, fix.val, color[:], fade_step, fix.val[:])
                    fades.append(fade)
            changed = True

        fades = update_fades(whale, fades, cycle=True)
        whale.flush()
        sleep(wave_speed)
    
    while fades:
        fades = update_fades(whale, fades, cycle=True)
        whale.flush()
        sleep(wave_speed)
    # print("=========== WAVE COMPLETE ===============")

def get_int(a:int, b:int):
    if a < 0:
        a = 0
    if a > 100:
        a = 100
    if b < 0:
        b = 0
    if b > 100:
        b = 100
    return randint(min(a,b), max(a,b))

def reset_lights(whale, fixtures):
    for fix in fixtures:
        fix.val = OFF[:]
        set_fixture(whale, fix, OFF)
    fades = []
    print('RESETTING LIGHTS')
    for light_array in lights['house']:
        for light in light_array:
            if light is not None:
                # print(light)
                light.val = OFF[:]
                set_fixture(whale, light, OFF)
    #             fades.append(
    #                 Fade(light, [100, 100, 100, 100], [0,0,0,0], -100, [100, 100, 100, 100]))
    # while fades:
    #     fades = update_fades(whale, fades, cycle=False)
    whale.flush()
        # sleep(0.3)
    print('RESET COMPLETE')
    # sleep(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="path to the config file")
    parser.add_argument('--on', action="store_true")
    parser.add_argument('--off', action="store_true")
    parser.add_argument('-r', '--refresh', help="Refresh rate in cycles per second, default is 30hz", default=30, type=int)
    parser.add_argument('-s', '--simulate', action="store_true", help="simulate the light values by printing to screen")
    args = parser.parse_args()

    input_queue = queue.Queue()
    input_thread = threading.Thread(target=add_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()
    wave_count = 0
    wave_max = 2

    with open(args.config, 'r') as instream:
        config = yaml.load(instream, yaml.SafeLoader)
        pprint(config)
    user = config['user']
    password = config['pass']
    
    queue_index = -1
    current_queue = config['queues'][queue_index]
    # color_pallet = current_queue['color_pallet']
    static_lights = True
    light_shape = (len(lights['house']), len(lights['house'][0]))


    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt32  # 32 bits per sample
    # channels = 1
    fs = 44100  # Record at 44100 samples per second
    # seconds = 1
    # cutoff = 10

    FRAMES = 10
    # FPS = 1.0/FRAMES

    VOL_THRESH = -40
    EDGE_THRESHOLD = 3

    volume_data = np.array([-100 for _ in range(FRAMES*5)])#np.zeroes((FRAMES * 5, 1))
    volume_idx = 0

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    stream = p.open(format=sample_format,
                    channels=1,
                    rate=fs,
                    input=True)
    fig, axs = plt.subplots(3)
    

    min_time_since_wave = timedelta(seconds=0.5)
    last_wave_time = datetime.now()

    wave_threads = []
    static_fixtures = [
        Fixture(423),
        Fixture(415),
        Fixture(419),
        Fixture(427)
    ]
    wave_speed = 0.1
    static = True
    try:
        with Whale(config['host'], config['port'], user=user, password=password, sim=args.simulate) as whale:

            reset_lights(whale, static_fixtures)
            refresh = timedelta(seconds=1.0/args.refresh)

            while True:
                # whale.sync()
                now = datetime.now()

                note = False
                print_lights(lights, light_shape)

                if not input_queue.empty():
                    
                    keys = input_queue.get()
                    print(f"input = {keys}")
                    if keys == 'q':
                        # reset everything
                        for wt in wave_threads:
                            wt.join()
                        print("New queue - resetting lights")
                        reset_lights(whale, static_fixtures)
                        sleep(0.2)

                        for fix in static_fixtures:
                            print(f"setting static fixture {fix.channel} to {OFF}")
                            set_fixture(whale, fix, OFF)
                        
                        queue_index += 1
    
                        print(f"Staring queue #{queue_index+1}")
                        try:
                            current_queue = config['queues'][queue_index]
                        except IndexError:
                            print("ITS DONE YO")
                            for thread in wave_threads:
                                thread.join()
                            reset_lights(whale, static_fixtures)
                            for fix in static_fixtures:
                                print(f"setting static fixture {fix.channel} to {OFF}")
                                set_fixture(whale, fix, OFF)
                            whale.flush()
                            sleep(2)
                            return 0
                        print(current_queue)
                        if 'color_pallet' in current_queue.keys():
                            static = False
                        else:
                            static = True
                

                        for fix in current_queue.get('light_values', []):
                            new_fix = Fixture(fix['patch'], 4, 0, 0)
                            new_fix.val = fix['values'][:]
                            static_fixtures.append(new_fix)
                            print(f"setting static fixture {new_fix.channel} to {new_fix.val}")
                            set_fixture(whale, new_fix, new_fix.val)
                        
                        if queue_index == 5:
                            for light_array in lights['house']:
                                for light in light_array:
                                    if light is not None:
                                        # print(light)
                                        light.val = [100, 100, 0, 10][:]
                                        set_fixture(whale, light, [100, 100, 0, 10])
                            # for i in range(521, 900, 4):
                            #     new_fix = Fixture(i, 4, 0, [100, 100, 0, 10])
                            #     set_fixture(whale, new_fix, new_fix.val)

                        whale.flush()
                            # print(new_fix)
                            # new_fade = Fade(new_fix, [0,0,0,0], fix['values'], 10, [0,0,0,0])
                            # fades.append(new_fade)
                            # static_fixtures.append(new_fix)
                            # print(f"Starting fade: {new_fade}")

                        # while fades:
                        #     for fade in fades:
                        #         fade.fixture.val
                        #         print(fade)
                            
                        #     fades = update_fades(whale, fades, cycle=False)
                        #     whale.flush()
                        #     # sleep(0.2)
                        #     sleep(1)
                        print("static fades complete")
                    
                

                note = False
                data = stream.read(chunk, exception_on_overflow = False)
                data = np.frombuffer(data, np.int32)
                segment = AudioSegment(
                    data.tobytes(), 
                    frame_rate=fs,
                    sample_width=data.dtype.itemsize, 
                    channels=1)
                volume = segment.dBFS
                # print(volume)
                
                
                if volume_idx < volume_data.shape[0]-1:
                    volume_data[volume_idx] = volume
                    volume_idx += 1
                else:
                    for idx in range(volume_data.shape[0]-1):
                        volume_data[idx] = volume_data[idx+1]
                    volume_data[volume_idx] = volume
                
                # if volume > VOL_THRESH and volume - volume_data[volume_idx-1] > EDGE_THRESHOLD:
                #     note = True
                window_size = 5
                if volume > VOL_THRESH and volume - EDGE_THRESHOLD > sum(volume_data[volume_idx-window_size:volume_idx])/window_size:
                    note = True
                    volume = int(volume)

                axs[0].plot(volume_data)
                w = np.fft.fft(data)
                freqs = np.fft.fftfreq(len(w))

                idx = np.argmax(np.abs(w))
                freq = freqs[idx]
                freq_in_hertz = abs(freq * fs)

                l = len(w)//2
                freqs_data = abs(w[:l - int(l/1.2)])

                # print(len(freqs_data))
                
                axs[1].plot(freqs_data, 'r') 
                axs[0].set_title(f"freq = {freq_in_hertz}, note = {note}")
                
                small = False
                if note:
                    axs[2].cla()
                    peak_indicies, props = signal.find_peaks(freqs_data, height=1e10, distance=5)
                    peak_heights = props['peak_heights']
                    max_peak = 0
                    if len(peak_heights) > 0:
                        max_peak = peak_indicies[np.argmax(peak_heights)]
                        axs[2].axvline(max_peak)
                        if max_peak > 20:
                            small = True
                    # for i, peak in enumerate(peak_indicies):
                        # freq = freqs_data[peak]
                        # freq_in_hertz = abs(freq * fs)
                        # peak_avg += freq_in_hertz
                        # axs[2].axvline(peak)

                        # magnitude = props["peak_heights"][i]
                        # print(f"{freq_in_hertz}hz with magnitude {magnitude:.3f}")
                        # print(f"{freq}hz with magnitude {magnitude:.3f}")
                    axs[2].plot(freqs_data, 'r')


                if not static and note and (datetime.now() - last_wave_time) > min_time_since_wave:
                    # whale.init_connection()
                    if small:
                        print("------ NEW SMALL ------")
                        # found = False
                        # while not found:
                        #     x = get_int(1, 8)
                        #     y = 1
                        #     if lights['house'][x][y] is not None:
                        #         found = True
                        source = (2, 5)
                        color = [100, 100, 100, 100]
                        wave_args = (
                            whale,
                            lights['house'])
                        kw_args = {
                            'shape':  light_shape,
                            'source': source,
                            'color':  color,
                            'wave_speed': 0.05,
                            'small': True
                        }
                        new_thread = threading.Thread(
                                target=wave, 
                                args=wave_args, 
                                kwargs=kw_args)
                        new_thread.start()
                    elif wave_count < wave_max:
                        wave_count += 1  
                        color = [
                            get_int(current_queue['color_pallet']['low'][0]-max_peak, current_queue['color_pallet']['low'][0]+max_peak),
                            get_int(current_queue['color_pallet']['low'][1], current_queue['color_pallet']['low'][1]),
                            get_int(current_queue['color_pallet']['low'][2]-max_peak, current_queue['color_pallet']['low'][2]+max_peak),
                            current_queue['color_pallet']['low'][3]
                        ]
                        
                        # color[3] += max(60 - abs(volume), 0)
                        # if color[3] > 100:
                        #     color[3] = 100

                        
                        source = (get_int(0, 3), get_int(0, 3))
                        # source = (0, 3) 
                        wave_args = (
                            whale,
                            lights['house'])
                        kw_args = {
                            'shape':  light_shape,
                            'source': source,
                            'color':  color,
                            'wave_speed': wave_speed
                        }
                        new_thread = threading.Thread(
                                target=wave, 
                                args=wave_args, 
                                kwargs=kw_args)
                        new_thread.start()
                        wave_threads.append(new_thread)
                        last_wave_time = datetime.now()
                        # print("NEW WAVE ADDED")
                    
                        

                time_delta = datetime.now() - now
                if time_delta < refresh:
                    # print((refresh - time_delta).total_seconds())
                    plt.pause((refresh - time_delta).total_seconds())
                    axs[0].cla()
                    axs[1].cla()

                wt_remove = []
                for wt in wave_threads:
                    if not wt.is_alive():
                        wt_remove.append(wt)
                for wt in wt_remove:
                    # print("REMOVING WAVE THREAD")
                    wave_threads.remove(wt)
                    wave_count -= 1
                print(f"vol: {volume}")
                print(f"total sent: {whale.total_sent}")
                # if wave_count:
                #     print(f"speed: {wave_speed}")
                    # sleep((refresh - time_delta).total_seconds())
    except KeyboardInterrupt:
        for thread in wave_threads:
            thread.join()
        reset_lights(whale, static_fixtures)
                
if __name__ == "__main__":
    main()
    
        