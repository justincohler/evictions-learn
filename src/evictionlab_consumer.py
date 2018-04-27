import boto3
import csv
from io import StringIO
from collections import deque

s3 = boto3.resource('s3')
bucket = s3.Bucket('eviction-lab-data-downloads')
# Iterates through all the objects, doing the pagination for you. Each obj
# is an ObjectSummary, so it doesn't contain the body. You'll need to call
# get to get the whole body

output_header=True
with open('C:/Users/Justin Cohler/output.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput, lineterminator='\n')

    for obj in bucket.objects.all():
        output = []

        state_header=True
        if 'block-groups.csv' not in obj.key:
            continue

        key = obj.key
        state=key[:2]

        if state == 'US':
            continue

        print("Writing " + state + ": " + key + "...")

        body = obj.get()['Body'].read()
        buff=StringIO(body.decode("utf-8"))
        reader = csv.reader(buff)

        for line in reader:
            if output_header == True:
                d = deque(line)
                d.appendleft("State")
                output_header = False
            elif state_header == True:
                state_header = False
                continue
            else:
                d = deque(line)
                d.appendleft(state)

            output.append(d)

        writer.writerows(output)
        print("Finished writing " + state + ": " + key)
