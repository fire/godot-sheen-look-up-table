import numpy as np
import os
import struct

def read_dds_r16f_to_list(input_filename, output_filename, width=128, height=128):
    PIXEL_COUNT = width * height
    DDS_HEADER_SIZE = 128

    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' was not found.")
        return

    try:
        with open(input_filename, 'rb') as f:
            
            f.seek(DDS_HEADER_SIZE)
            
            data_bytes = f.read(PIXEL_COUNT * 2) 
        
        # float16
        data_half_precision = np.frombuffer(data_bytes, dtype=np.float16)

        # float16 to float32
        data_full_precision = data_half_precision.astype(np.float32)

        # numpy array to Python list
        float_list = data_full_precision.tolist()

        with open(output_filename, 'w') as out_f:
            out_f.write(str(float_list))
            
        print(f"Success: {len(float_list)} values ​​have been extracted and saved to '{output_filename}'.")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

input_file = 'sheen_lut.dds'
output_file = 'sheen_lut_data.txt'

read_dds_r16f_to_list(input_file, output_file)
