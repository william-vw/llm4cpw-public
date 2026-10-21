def jaccard_simil(s1, s2):
    inters = s1.intersection(s2)
    union = s1.union(s2)
    return len(inters) / len(union)