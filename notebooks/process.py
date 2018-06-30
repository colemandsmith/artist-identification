import csv
import os
import zipfile
import subprocess
from os.path import expanduser

data_dir = expanduser("~") + "/Data/"
all_artist_data = data_dir + "all_artist_data.csv"
filtered = data_dir + "filtered.csv"

def main():
    count_map = count()
    filter_at_threshold(300, count_map)

def count():

    #trim the irrelevant files; make a csv of our subset of the data

    count_map = {}

    with open(all_artist_data, 'r') as to_read:
        files = os.listdir(data_dir + "train")
        reader = csv.reader(to_read, quotechar='\"')
        header = next(reader)
        for row in reader:
            if row[-1] in files:
                if row[0] in count_map:
                    count_map[row[0]] += 1
                else:
                    count_map[row[0]] = 1
                
    return count_map

def filter_at_threshold(threshold, count_map):
    count = 0
    not_here = 0
    to_filter=set()
    with open(filtered, "w") as to_write, open(all_artist_data, 'r') as to_read:
        
        writer = csv.writer(to_write, delimiter=",",quotechar='\"')
        files = os.listdir("train")
        print(str(len(files)) + " files")
        reader = csv.reader(to_read, quotechar='\"')
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            if row[-1] in files and count_map[row[0]] > threshold:
                count +=1
                writer.writerow(row)
            elif row[-1] not in files:
                #print("wtf")
                not_here +=1
            else:
                to_filter.add('train/' + row[-1])
    print(str(len(to_filter)) + " to filter")
    print(str(not_here) + " not here")
    for f in to_filter:
        subprocess.call(['rm', f])
    return to_filter

if __name__ == "__main__":
    main()
