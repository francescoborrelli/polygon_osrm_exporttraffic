from ws_maps.pgrouting import PGRouting
pgr = PGRouting(user='postgres', password='postgres', host='localhost', port=5439, database='osm2podb')
#f = open('demofile.csv', "w")
#pgr.edges_to_csv('demofile3.csv', ['osm_source_id', 'osm_target_id','osm_lat'])
#pgr.edges_to_csv('demofile2.csv', ['osm_target_id','osm_source_id'])
#pgr.edges_to_csv('demofile.csv', ['osm_id', 'km', 'kmh', 'osm_source_id', 'osm_target_id', 'cost', 'reverse_cost'])
#pgr.edges_to_csv('demofile.csv', ['osm_id', 'km', 'kmh', 'osm_source_id', 'osm_target_id', 'cost', 'reverse_cost'])

from csv import writer
from csv import reader
default_text = '30'
# Open the input_file in read mode and output_file in write mode
with open('demofile.csv', 'r') as read_obj, \
        open('demofilefinal.csv', 'w', newline='') as write_obj:
    # Create a csv.reader object from the input file object
    csv_reader = reader(read_obj)
    # Create a csv.writer object from the output file object
    csv_writer = writer(write_obj)
    # Read each row of the input csv file as list
    next(csv_reader)
    for row in csv_reader:
        # Append the default text in the row / list
        row.append(default_text)
        row.append(default_text)
        #row.append('')
        # Add the updated row / list to the output file
        csv_writer.writerow(row)
if 0:
    with open('demofile2.csv', 'r') as read_obj, \
            open('demofilefinal.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_reader = reader(read_obj)
        # Read each row of the input csv file as list
        next(csv_reader)
        for row in csv_reader:
            # Append the default text in the row / list
            row.append(default_text)
            row.append(default_text)
            #row.append('')
            # Add the updated row / list to the output file
            csv_writer.writerow(row)
