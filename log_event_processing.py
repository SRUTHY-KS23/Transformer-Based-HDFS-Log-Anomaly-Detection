import re
import pickle
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm 
from config import LOG_FILE, LABEL_FILE, SAMPLE_SIZE
from spell import LogParser as SpellParser
from drain import LogParser as DrainParser

def parse_and_encode_logs(parser_type="drain"):
    
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    rex = [r'blk_-?\d+', r'(\d+\.){3}\d+', r'\d+']
    
    if parser_type.lower() == "spell":
        print("Using SPELL Engine...")
        parser = SpellParser(indir='./', outdir='./result/', log_format=log_format, tau=0.5, rex=rex)
        template_col = 'Message'
    else:
        print("Using DRAIN Engine...")
        parser = DrainParser(log_format=log_format, indir='./', outdir='./result/', rex=rex)
        template_col = 'EventTemplate'

    parser.parse(LOG_FILE)
  
    struct_path = f'./result/{os.path.basename(LOG_FILE)}_structured.csv' if parser_type == "drain" else './result/out_structured.csv'
    structured_df = pd.read_csv(struct_path)

  
    rows = []
  
    for i, row in tqdm(structured_df.iterrows(), total=len(structured_df), desc="Mapping Blocks"):
        if i >= SAMPLE_SIZE: break
        
      
        blk = re.search(r"(blk_[-0-9]+)", str(row['Content']))
        if blk:
            rows.append({
                "BlockID": blk.group(1), 
                "Template": row[template_col]
            })

    df = pd.DataFrame(rows)

    print("Encoding templates...")
    le = LabelEncoder()

    df["EventId"] = le.fit_transform(df["Template"]) + 1

    vocab_size = len(le.classes_) + 1
    pickle.dump(le, open("label_encoder.pkl", "wb"))
    with open("vocab_config.pkl", "wb") as f:
        pickle.dump({"vocab_size": vocab_size}, f)

    print("Grouping logs into sequences by BlockID...")
    sequences = df.groupby("BlockID")["EventId"].apply(list).reset_index()
    print("Merging sequences with Anomaly labels...")
    if os.path.exists(LABEL_FILE):
        labels_df = pd.read_csv(LABEL_FILE)
        final_df = pd.merge(sequences, labels_df, left_on="BlockID", right_on="BlockId", how="inner")
        final_df['Label'] = final_df['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
        final_df[['BlockID', 'EventId', 'Label']].to_csv("labeled_sequences.csv", index=False)
        print(f"Success! Created labeled_sequences.csv with {len(final_df)} blocks.")
    else:
        print("Warning: LABEL_FILE not found. Saving sequences without labels.")
        sequences.to_csv("hdfs_sequences.csv", index=False)
    event_map = pd.DataFrame({
        "EventId": range(1, vocab_size), 
        "EventTemplate": le.classes_
    })
    event_map.to_csv("event_mapping.csv", index=False)

    print(f"Final Vocab Size: {vocab_size}")
    return sequences

if __name__ == "__main__":
    parse_and_encode_logs(parser_type="drain")