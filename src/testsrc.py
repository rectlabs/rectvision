from rectvision import data, models, processors, settings
from rectvision.data import converters

converters.LabelmeToTfCsv(ann_dir='/valid', out_csv_dir='/cocofiles')
converters.RectvisionConverter(0.6, 0.2, 0.2, 'yolo-txt')