import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import subprocess
import os
import json


class NER():
    def __init__(self, batch_size, project_name, project_dir, optimize):
        self.batch_size = batch_size
        self.project_name = project_name
        self.optimize = optimize
        self.project_dir = self.valid_path(project_dir)
        self.train_annotations = os.path.join(self.project_dir,'data/training_data.json')
        self.test_annotations = os.path.join(self.project_dir,'data/testing_data.json')
        self.nlp = spacy.blank("en") # load a new spacy model
        self.db = DocBin() # create a DocBin object
    def valid_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        return path

    def convert_json(self,data_path):
        data_json = json.load(open(data_path))
        for text, annot in tqdm(data_json['annotations']): 
            doc = self.nlp.make_doc(text) 
            ents = []
            for start, end, label in annot["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    print("Skipping entity")
                else:
                    ents.append(span)
            doc.ents = ents 
            self.db.add(doc)
        file_name = data_path.split('/')[-1].split('.')[0]+'.spacy'
        file_path = os.path.join(self.project_dir,file_name)
        self.db.to_disk(file_path) # save the docbin object

    def create_data_config(self):
        subprocess.call(["Python",  
                        "-m",
                        "spacy",
                        "init",
                        "config", os.path.join(self.project_dir,"config.cfg"),
                        "--lang", "en",
                        "--pipeline", "ner", 
                        "--optimize", self.optimize,"-F"])   

    def train(self):
        # self.setup()
        self.convert_json(self.train_annotations)
        train_path = os.path.join(self.project_dir,'training_data.spacy')
        if os.path.exists(self.test_annotations):
            self.convert_json(self.test_annotations)
            test_path = os.path.join(self.project_dir,'testing_data.spacy')
        else:
            test_path = train_path

        self.create_data_config()
        

        subprocess.call(["Python", 
                        "-m",
                        "spacy",
                        "train", os.path.join(self.project_dir,"config.cfg"),
                        "--output", os.path.join(self.project_dir,"output"),
                        "--paths.train", train_path,
                        "--paths.dev", test_path,
                        "--nlp.batch_size", str(self.batch_size)])
    
    def get_entites(self, text):
        model_path = os.path.join(self.project_dir,"output","model-best")
        nlp_ner = spacy.load(model_path)
        doc = nlp_ner(text)
        return doc
                        
    
# project_dir = r"C:\Users\USER\yoruba-net\ocr"
# batch_size, project_name, project_dir, optimize = 32, "test_project", project_dir, 'efficiency'
# model = NER(batch_size, project_name, project_dir, optimize)
# text = "As for major cryptocurrencies, Bitcoin tumbled 2.46 percent to trade at Rs 37,49,173 while Ethereum fell 4.48 percent at Rs 2,93,527.4. Cardano declined 7.25 percent to Rs 105.4. Avalanche fell 8.77 percent to Rs 8,048, Polkadot tumbled 7.53 percent at Rs 2,137.04 and Litecoin dipped 1.04 percent to Rs 11,725.33 over the last 24 hours. Tether rose 0.12 percent to trade at Rs 80.18."
# # model.train()
# doc = model.get_entites(text)
# for ents in doc.ents:
#     print(ents.text,"|",ents.label_, "|", ents.start_char, "|", ents.end_char)