#!/bin/python
import subprocess
import time
import sys

def printq(arg):
    if not quiet:
        print(arg)

quiet = False
period = 0.5  # default sampling period in seconds
duration = 10  # default total duration in seconds
for opt in sys.argv[1:]:
    if opt in ('help', '--help', '-h'):
        print('Usage: %s [-q] [<period>] [<duration>]' % sys.argv[0])
        print('Prints average power consumption for the Raspberry Pi over a duration.\n')
        print('-q quiet output (only print average total power consumption)')
        sys.exit(0)
    elif opt == '-q':
        quiet = True
    elif period == 0:
        period = float(opt)
    else:
        duration = float(opt)

samples = []
start_time = time.time()

while time.time() - start_time < duration:
    result = subprocess.run(['vcgencmd', 'pmic_read_adc'], stdout=subprocess.PIPE)
    lines = result.stdout.decode('utf-8').split('\n')
    curs = {}
    vols = {}
    pows = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        label = line.split(' ')[0]
        val = float(line.split('=')[1][:-1])
        shortLabel = label[:-2]
        if label[-1] == 'A':
            curs[shortLabel] = val
        else:
            vols[shortLabel] = val
    for key in list(curs.keys()):
        if key in vols:
            pows[key] = curs[key] * vols[key]
            printq('%10s %10f V %10f A %10f W' % (key, vols[key], curs[key], pows[key]))
            del curs[key]
            del vols[key]
    for key in vols:
        printq('%10s %10f V ' % (key, vols[key]))
    for key in curs:
        printq('%10s            %10f A ' % (key, curs[key]))
    
    total_power = sum(pows.values())
    samples.append(total_power)

    if not quiet:
        print('            Instantaneous total power: %10f W' % total_power)

    time.sleep(period)

avg_power = sum(samples) / len(samples)
print('\nAverage power over %.1f seconds (%d samples): %.5f W' % (duration, len(samples), avg_power))
