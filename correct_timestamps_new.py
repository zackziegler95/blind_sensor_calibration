import glob
import os

input_dir = 'data_new'
output_dir = 'data_tscorrect_new'

num_sensors = 8

#plt.figure()

for i in range(1, num_sensors+1):
    fname = glob.glob(f'{input_dir}/Sensor{i}*.csv')[0]
    basename = os.path.basename(fname)
    with open(f'{output_dir}/{basename}', 'w') as fw:
        with open(f'{input_dir}/{basename}', 'r') as fr:
            for i, line in enumerate(fr):
                if i > 0:
                    parts = line.strip().split(',')
                    time = parts[1]
                    time_parts = time.split(' ')
                    day = time_parts[0].split('-')[2]
                    
                    if day == '15':
                        time_parts.append('PM')
                    elif day == '16':
                        time_parts.append('AM')

                    parts[1] = ' '.join(time_parts)
                    line = ','.join(parts)+'\n'
                fw.write(line)
