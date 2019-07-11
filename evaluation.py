"""
Outline:
    load model
    create dataset
        somewhere there needs to be a sliding window thing, unclear where
    push an image through with all sliding windows
    collect all the heat maps
        construct them with a measure of abnormality, probably try several
        data structure could be one giant tensor
    combine the heat maps
    compare with labels
    
    
    
Notes:
    - write everything to be [BxCxHxW]-compatible. I might want to do it on the cluster.