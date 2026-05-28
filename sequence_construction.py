import pandas as pd
import numpy as np
import pickle
import os
import time
from tqdm import tqdm
from sklearn.utils import resample
class ResearchConfig:
    SEQ_LEN = 100               
    STEP_SIZE = 20              
    MIN_EVENTS = 3              
    TEST_RATIO = 0.2            
    RANDOM_SEED = 42
    INPUT_PARSED = "parsing_result.csv"   
    INPUT_MAPPING = "event_mapping.csv"   
    INPUT_LABELS = "/kaggle/input/sruthyksme/anomaly_label.csv" 
    
    OUTPUT_DIR = "./transformer_ready_data/"
    CSV_FINAL = "final_research_sequences.csv"
    
    RESERVE_TOKENS = 2         
    OVERSAMPLE_ANOMALIES = True 

if not os.path.exists(ResearchConfig.OUTPUT_DIR):
    os.makedirs(ResearchConfig.OUTPUT_DIR)
def sliding_window_processor(event_list, window_size, step):
    if len(event_list) < window_size:
        return [event_list]
    windows = []
    for i in range(0, len(event_list) - window_size + 1, step):
        windows.append(event_list[i : i + window_size])
    if (len(event_list) - window_size) % step != 0:
        windows.append(event_list[-window_size:])
    return windows

def pad_sequence_to_fixed(seq, max_len):
    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))
    return seq[:max_len]
class SequenceEngine:
    def __init__(self):
        self.conf = ResearchConfig()
        
    def load_assets(self):
        print("📥 Loading and Merging assets...")
    
        parsed_df = pd.read_csv(self.conf.INPUT_PARSED)
        mapping_df = pd.read_csv(self.conf.INPUT_MAPPING)
        
        parsed_df = pd.merge(parsed_df, mapping_df, on="Content", how="left")
        
        labels_df = pd.read_csv(self.conf.INPUT_LABELS)
        
        labels_df.columns = labels_df.columns.str.strip()
        labels_df['BlockId'] = labels_df['BlockId'].astype(str).str.strip()
        parsed_df['BlockID'] = parsed_df['BlockID'].astype(str).str.strip()
       
        labels_df['Label'] = labels_df['Label'].str.strip().str.capitalize().map({'Normal': 0, 'Anomaly': 1})
        
        parsed_df['EventId'] = parsed_df['EventId'].fillna(1) 
        parsed_df['EventId'] = parsed_df['EventId'].astype(int) + self.conf.RESERVE_TOKENS
        
        return parsed_df, labels_df

    def execute_pipeline(self):
        parsed_df, labels_df = self.load_assets()


        print("🔗 Grouping events by BlockID...")
        tqdm.pandas(desc="Grouping")
        grouped = parsed_df.groupby("BlockID")["EventId"].progress_apply(list).reset_index()

        print("🧬 Syncing labels...")
        merged = pd.merge(grouped, labels_df, left_on="BlockID", right_on="BlockId", how="inner")

        print(f"✂️ Windowing (Stride: {self.conf.STEP_SIZE})...")
        final_data = []
        for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Windowing"):
            if len(row['EventId']) < self.conf.MIN_EVENTS:
                continue
            windows = sliding_window_processor(row['EventId'], self.conf.SEQ_LEN, self.conf.STEP_SIZE)
            for w in windows:
                final_data.append({
                    "BlockID": row['BlockID'],
                    "Sequence": pad_sequence_to_fixed(w, self.conf.SEQ_LEN),
                    "Label": row['Label']
                })

        processed_df = pd.DataFrame(final_data)

        if self.conf.OVERSAMPLE_ANOMALIES:
            print("⚖️ Balancing dataset...")
            normal_set = processed_df[processed_df.Label == 0]
            anomaly_set = processed_df[processed_df.Label == 1]
            if not anomaly_set.empty:
                anomaly_upsampled = resample(anomaly_set, replace=True, n_samples=len(normal_set), random_state=42)
                processed_df = pd.concat([normal_set, anomaly_upsampled]).sample(frac=1).reset_index(drop=True)

        
        csv_path = os.path.join(self.conf.OUTPUT_DIR, self.conf.CSV_FINAL)
        viewable_df = processed_df.copy()
        viewable_df['Sequence'] = viewable_df['Sequence'].apply(lambda x: " ".join(map(str, x)))
        viewable_df.to_csv(csv_path, index=False)

        X = np.array(processed_df['Sequence'].tolist())
        y = processed_df['Label'].values
        vocab_size = int(parsed_df['EventId'].max() + 1)

        with open(os.path.join(self.conf.OUTPUT_DIR, "transformer_input.pkl"), "wb") as f:
            pickle.dump({"X": X, "y": y, "vocab_size": vocab_size}, f)

        print(f"✅ DONE! Generated {len(processed_df)} sequences for Training.")

if __name__ == "__main__":
    engine = SequenceEngine()
    engine.execute_pipeline()