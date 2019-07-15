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
"""

model.eval()
data_provider takes slice?
    But then I don't yet know how big the image is, that doesn't make sense
    Aim: fix step_size
    But images have variable size
    What if it isn't cleanly divideable?
    
    
    
    could use an iter object
    the thing is, if I want to use the dataloader (probably a good idea if I want to do it on the GPU), then __getitem__() can only take index as parameters...
        If I want that to work, I would have to create a new dataloader for EACH IMAGE
        Prolly not, I could just return input, target, image_index, slice_location
    
probably something with slice
outputs = model.forward(inputs)
measure of abnormality between outputs and targets
incorporate that into big ass tensor
