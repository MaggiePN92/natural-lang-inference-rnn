def check_dataloader(dataloder):
    print(f"Length of dataloader: {len(dataloder)}")

    for premise, hypothesis, targets in dataloder:
        print(f"Premise shape: {premise.shape}")
        print(f"Hypothesis shape: {hypothesis.shape}")
        print(f"Targets shape: {targets.shape}")
        break
