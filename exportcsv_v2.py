#from ws_maps.pgrouting import PGRouting
#pgr = PGRouting(user='postgres', password='postgres', host='localhost', port=5439, database='osm2podb')
#f = open('demofile.csv', "w")
#pgr.edges_to_csv('demofile3.csv', ['osm_source_id', 'osm_target_id','osm_lat'])
#pgr.edges_to_csv('demofile2.csv', ['osm_target_id','osm_source_id'])
#pgr.edges_to_csv('demofile.csv', ['osm_id', 'km', 'kmh', 'osm_source_id', 'osm_target_id', 'cost', 'reverse_cost'])
#pgr.edges_to_csv('demofile.csv', ['osm_id', 'km', 'kmh', 'osm_source_id', 'osm_target_id', 'cost', 'reverse_cost'])

from csv import writer
from csv import reader
import numpy as np
default_text = '6'
# Open the input_file in read mode and output_file in write mode
with open('all_osm_edges.csv', 'r') as read_obj, \
        open('updates_all_edges.csv', 'w', newline='') as write_obj:
    # Create a csv.reader object from the input file object
    csv_reader = reader(read_obj)
    # Create a csv.writer object from the output file object
    csv_writer = writer(write_obj)
    # Read each row of the input csv file as list
    next(csv_reader)
    for row in csv_reader:
        # Append the default text in the row / list
        reversecost=float(row[8])
        if len(row[-2][1:-1])>1:
            nodelist=[int(s) for s in row[-2][1:-1].split(',')]
            if len(nodelist)>1:
                for k in range(len(nodelist)-1):
                    newrow=[]
                    newrow.append(str(nodelist[k]))
                    newrow.append(str(nodelist[k+1]))
                    newrow.append(default_text)
                    newrow.append(default_text)
                    #row.append('')
                    # Add the updated row / list to the output file
                    csv_writer.writerow(newrow)
                if reversecost<1e6:
                    for k in range(len(nodelist) - 1):
                        newrow = []
                        newrow.append(str(nodelist[k+1]))
                        newrow.append(str(nodelist[k]))
                        newrow.append(default_text)
                        newrow.append(default_text)
                        # row.append('')
                        # Add the updated row / list to the output file
                        csv_writer.writerow(newrow)
            else:
                newrow = []
                newrow.append(row[3])
                newrow.append(row[4])
                newrow.append(default_text)
                newrow.append(default_text)
                csv_writer.writerow(newrow)
                newrow = []
                newrow.append(row[4])
                newrow.append(row[3])
                newrow.append(default_text)
                newrow.append(default_text)
                csv_writer.writerow(newrow)
        else:
            newrow = []
            newrow.append(row[3])
            newrow.append(row[4])
            newrow.append(default_text)
            newrow.append(default_text)
            csv_writer.writerow(newrow)
            newrow = []
            newrow.append(row[4])
            newrow.append(row[3])
            newrow.append(default_text)
            newrow.append(default_text)
            csv_writer.writerow(newrow)

