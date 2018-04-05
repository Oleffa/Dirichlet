def normalize(raw):
    out = [float(i)/sum(raw) for i in raw]
    return out