import torch


def text_pad(text, period):
    if len(text) > period:
        return text[-period:]
    elif len(text) == period:
        return text
    else:
        pad=[]
        pad.extend(text)
        for i in range(period-len(text)):
            pad.append("")
        return pad

def created_pad(created,period):
    data=torch.tensor(created, dtype=torch.long)
    if len(created) > period:
        data=data[-period:]
        first_period=data[0].clone()
        data-=first_period
        return data
    elif len(created) == period:
        return data
    else:
        pad=torch.cat((data,torch.zeros(period-len(created))))
        return pad.to(torch.long)

def make_pad(text,period):
    if len(text) >= period:
        return torch.ones(period)
    elif len(text) < period:
        return torch.cat((torch.ones(len(text)),torch.zeros(period-len(text))))

class to_padding():
    def __init__(self, text, created, period):
        self.text = text
        self.created = created
        self.period = period
    
    def pad_data(self):
        text = [text_pad(i, self.period) for i in self.text]
        created = torch.stack([created_pad(i, self.period) for i in self.created])
        padding = torch.stack([make_pad(i, self.period) for i in self.text])
        return text, created, padding

    def pad_text(self):
        text = [text_pad(i, self.period) for i in self.text]
        return text

    def pad_created(self):
        created = torch.stack([created_pad(i, self.period) for i in self.created])
        return created

    def make_padding(self):
        padding = torch.stack([make_pad(i, self.period) for i in self.text])
        return padding

