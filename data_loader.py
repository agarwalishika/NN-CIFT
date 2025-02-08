from datasets import load_dataset

class Data:
    def __init__(self, existing_data_name, new_data_name, length):
        dataloader = Dataloader(length)
        self.existing_prompts, self.existing_references, self.existing_data = dataloader.get_existing(existing_data_name)

        dname = ""
        if new_data_name is None:
            dname = existing_data_name
            self.dataset = f"{existing_data_name[existing_data_name.rfind('/')+1:]}|{existing_data_name[existing_data_name.rfind('/')+1:]}"
        else:
            dname = new_data_name
            self.dataset = f"{existing_data_name[existing_data_name.rfind('/')+1:]}|{new_data_name[new_data_name.rfind('/')+1:]}"

        self.new_prompts, self.new_references, self.new_data = dataloader.get_new(dname)
        self.test_prompts, self.test_references, self.test_data = dataloader.get_test(dname)

class Dataloader:
    def __init__(self, length) -> None:
        self.length = length
    
    def get_existing(self, name):
        if "mix-instruct" in name:
            return self.get_mixinstruct_train()
        if "hotpot" in name:
            return self.get_hotpot_train()
        if "alpaca" in name:
            return self.get_alpaca_train()
        0/0
    
    def get_new(self, name):
        if "mix-instruct" in name:
            return self.get_mixinstruct_valid()
        if "hotpot" in name:
            return self.get_hotpot_valid()
        if "alpaca" in name:
            return self.get_alpaca_valid()
        0/0
    
    def get_test(self, name):
        if "mix-instruct" in name:
            return self.get_mixinstruct_test()
        if "hotpot" in name:
            return self.get_hotpot_test()
        if "alpaca" in name:
            return self.get_alpaca_test()
        0/0
    
    ############ MIX-INSTRUCT ###############
    def get_mixinstruct(self, split):
        ds = load_dataset("llm-blender/mix-instruct")[split].to_pandas()[:self.length]
        prompts = ds['instruction'] + " " + ds['input']
        references = ds['output']
        data = prompts + "\n##Answer: " + references
        return list(prompts), list(references), list(data)

    def get_mixinstruct_train(self):
        return self.get_mixinstruct('train')
    
    def get_mixinstruct_valid(self):
        return self.get_mixinstruct('validation')
    
    def get_mixinstruct_test(self):
        return self.get_mixinstruct('test')
    
    ########## HOTPOT QA ###################
    def get_hotpot(self, split):
        ds = load_dataset('hotpotqa/hotpot_qa', 'fullwiki', trust_remote_code=True)[split].to_pandas()
        data, prompts, references = [], [], []
        for _, i in ds.iterrows():
            instruction = i['question']
            output = i['answer']

            input = ""
            temp = i['supporting_facts']
            keep = True
            for x in range(len(temp['title'])):
                try:
                    ind = list(i['context']['title']).index(temp['title'][x])
                    input += i['context']['sentences'][ind][temp['sent_id'][x]] + " "
                except:
                    hi = 9

            if len(input) > 0:
                prompts.append(f"{instruction} {input}")
                references.append(output)
                data.append(f"{instruction} {input}\n##Answer: {output}")
            
            if len(data) >= self.length:
                return list(prompts), list(references), list(data)
        return list(prompts), list(references), list(data)
    
    def get_hotpot_train(self):
        return self.get_hotpot('train')
    
    def get_hotpot_valid(self):
        return self.get_hotpot('validation')
    
    def get_hotpot_test(self):
        og_len = self.length
        self.length *= 2
        p, r, d = self.get_hotpot('validation')
        self.length = og_len
        return p[og_len:], r[og_len:], d[og_len:]
    
    
    ############   ALPACA    ###############
    def get_alpaca(self, split):
        ds = load_dataset("tatsu-lab/alpaca")['train'].to_pandas()

        if split == 'train':
            ds = ds[:self.length]
        elif split == 'valid':
            ds = ds[self.length:2*self.length]
        else:
            ds = ds[2*self.length:3*self.length]
        
        prompts = ds['instruction'] + " " + ds['input']
        references = ds['output']
        data = prompts + "\n##Answer: " + references
        return list(prompts), list(references), list(data)

    def get_alpaca_train(self):
        return self.get_alpaca('train')
    
    def get_alpaca_valid(self):
        return self.get_alpaca('valid')
    
    def get_alpaca_test(self):
        return self.get_alpaca('test')
