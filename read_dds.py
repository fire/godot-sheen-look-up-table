import struct
import numpy as np

def read_dds_rgba16f(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(128)
        
        magic = header[:4]
        if magic != b'DDS ':
            raise ValueError("Invalid DDS")
        
        width = 128
        height = 128
        channels = 4
        bytes_per_channel = 2
        
        total_bytes = width * height * channels * bytes_per_channel
        pixel_data = f.read(total_bytes)
        
        if len(pixel_data) < total_bytes:
            raise ValueError(f"Incomplete file. Expected {total_bytes} bytes, got {len(pixel_data)}")
    
    return pixel_data, width, height

def extract_blue_channel(pixel_data, width, height):
    blue_values = []
    channels = 4
    bytes_per_channel = 2
    bytes_per_pixel = channels * bytes_per_channel
    
    for i in range(width * height):
        offset = i * bytes_per_pixel
        
        blue_offset = offset + (2 * bytes_per_channel)
        
        blue_bytes = pixel_data[blue_offset:blue_offset + bytes_per_channel]
        
        blue_value = np.frombuffer(blue_bytes, dtype=np.float16)[0]
        blue_values.append(float(blue_value))
    
    return blue_values

def save_to_txt(values, output_filepath):
    with open(output_filepath, 'w') as f:
        f.write('[')
        for i, value in enumerate(values):
            if i > 0:
                f.write(', ')
            if i % 10 == 0 and i > 0:
                f.write('\n ')
            f.write(f'{value}')
        f.write(']')
    
    print(f"{len(values)} values ​​were saved in {output_filepath}")

if __name__ == "__main__":
    input_dds = "dfg_lut.dds"
    output_txt = "sheen_lut_data.txt"
    
    try:
        print(f"Reading {input_dds}...")
        pixel_data, width, height = read_dds_rgba16f(input_dds)
        
        print("Extracting Blue channel...")
        blue_values = extract_blue_channel(pixel_data, width, height)
        
        print(f"Saving values ​​in {output_txt}...")
        save_to_txt(blue_values, output_txt)
        
    except FileNotFoundError:
        print(f"Error: File not found {input_dds}")
    except Exception as e:
        print(f"Error: {e}")
