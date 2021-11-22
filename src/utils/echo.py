import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)
import random
from scipy.signal import fftconvolve
import numpy as np

def get_random_room():    
    # The desired reverberation time and dimensions of the room
    length, width, height = random.uniform(35,50),random.uniform(12,30),random.uniform(15,28)
    
    room_dim = [length, width, height]  # meters
    time = random.uniform(1.5, 4)
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(time, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=44100, materials=pra.Material(e_absorption), max_order=max_order
    )
    
    dir_obj = CardioidFamily(
        orientation=DirectionVector(azimuth=90, colatitude=15, degrees=True),
        pattern_enum=DirectivityPattern.HYPERCARDIOID,
    )
    
    mic_x, mic_y, mic_z = random.uniform(0, length), random.uniform(0, width), random.uniform(0, height)
    
    
    room.add_source(position=[random.uniform(2, 5), width / 2, random.uniform(.9, 1.3)], directivity=dir_obj)
    room.add_microphone(loc=[mic_x, mic_y, mic_z], directivity=dir_obj)
    
    return room

def get_room_impulse(room):
    room.compute_rir()
    ir = room.rir[0][0]
    return np.array(ir)

def convolve_with_room(x, ir):
    return fftconvolve(x, ir)

def get_room_irs(num_rooms=50):
    print("getting irs")
    
    rooms = []
    for i in range(num_rooms):
        room = get_random_room()
        ir = get_room_impulse(room)
        rooms.append(ir)
    return np.array(rooms)