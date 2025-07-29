def create_messages(row, mode="train"):
    """
    Create a message template for training.
    Adds system, user (with image and text), and assistant (with label) roles to the row.
    """
    row["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": row['image']},
            {"type": "text", "text": row['text']}
        ]},
        
    ]
    if mode == "train":
        row["messages"].append({"role": "assistant", "content": [{"type": "text", "text": row['label']}]})
        
    return row
 

def create_prefix_messages(dataset, prefix):
    """
    Add a prefix text to the user message content for each sample in the dataset.
    The prefix is prepended to each text item in the user message.
    """  
    prefix = "".join(prefix) if isinstance(prefix, list) else prefix
    new_dataset = []
    for i in range(len(dataset)):
        data = dataset[i]
        content = data['messages'][1]['content']
        new_content = []
        new_content.append({"type": "text", "text": prefix})
        for item in content:
            if item["type"] == "image":
                new_content.append(item)
            elif item["type"] == "text":
                new_content.append({"type": "text", "text": prefix + item["text"]})
        data['messages'][1]['content'] = new_content
        new_dataset.append(data)
    return new_dataset