




def set_model_mode(mode, models):
    if mode == "train":
        for model in models:
            model.train()
    elif mode == "eval":
        for model in models:
            model.eval()
    else:
        raise ValueError

