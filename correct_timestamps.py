# It seems like there was an error with the data writing software, so am/pm didn't get
# written to the data files properly. Luckly, it's easy to fix this manually because there
# is only one day crossing

input_dir = 'data'
output_dir = 'data_tscorrect'

num_sensors = 8

for i in range(1, num_sensors+1):
    with open(f'{output_dir}/Sensor{i}.csv', 'w') as fw:
        with open(f'{input_dir}/Sensor{i}.csv', 'r') as fr:
            for i, line in enumerate(fr):
                if i > 0:
                    parts = line.strip().split(',')
                    time = parts[1]
                    time_parts = time.split(' ')
                    day = time_parts[0].split('-')[2]
                    
                    # just look at the day to determine if the reading should have been am/pm
                    if day == '11':
                        time_parts.append('PM')
                    elif day == '12':
                        time_parts.append('AM')

                    parts[1] = ' '.join(time_parts)
                    line = ','.join(parts)+'\n'
                fw.write(line)
