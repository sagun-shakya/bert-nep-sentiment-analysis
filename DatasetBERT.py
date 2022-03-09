from torch.utils.data import Dataset
from numpy import asarray

class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, train_type):

        choice_train_types = ['concat', 'non_concat', 'text']
        assert train_type in choice_train_types, f'Train type should be one of {choice_train_types}.'

        self.labels = df['polarity'].tolist()
        self.ac_size = df['ac'].nunique()
        
        if train_type == 'text':
            self.texts = [tokenizer(text, 
                                    padding='max_length', max_length = 512, truncation=True,
                                    return_tensors="pt") for text in df['text']]

        else:
            # Concatenating the aspect categories to the text features.
            df['text'] = df['text'] + " " + df["ac"]
            
            # If concat, pass on the aspect terms to the tokenizer. 
            if train_type == 'concat':
                texts = []
                for ii in range(len(df)):
                    text, at = df[['text', 'at']].iloc[ii].to_list()
                    res = tokenizer(text, at, padding="max_length", max_length = 512, truncation=True, return_tensors = "pt")
                    texts.append(res)
            
            # If not, pass on the text plus aspect categories.
            elif train_type == 'non_concat':
                texts = [tokenizer(text, 
                                    padding='max_length', max_length = 512, truncation=True,
                                    return_tensors="pt") for text in df['text']]
            
            self.texts = texts
        

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return asarray(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y