import torch

def frombytes(data):
    for d in data:
        yield torch.frombuffer(d, dtype=torch.uint8)

def chop(data, seqlen=1024, dim=0):
    for d in data:
        offset = torch.randint(0, seqlen, ())
        for o in range(offset, d.shape[dim] - seqlen, seqlen):
            yield d.transpose(dim, 0)[o:o+seqlen].transpose(dim, 0)

def shuffle(data, bufsize=2048):
    buffer = [None] * bufsize
    for d in data:
        i = torch.randint(0, bufsize, ())
        out = buffer[i]
        buffer[i] = d
        if out is not None:
            yield out
    for out in buffer:
        if out is not None:
            yield out

def stack(data, batch):
    buffer = []
    for d in data:
        buffer += [d]
        if len(buffer) >= batch:
            yield torch.stack(buffer, dim=0)
            buffer = []
