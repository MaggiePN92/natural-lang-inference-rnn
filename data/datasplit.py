from sklearn.model_selection import train_test_split
from typing import List


def stratified_split(
        targets : List[str], premise : List[str], hypo : List[str],
        test_size=10_000
    ):
    targets_train, targets_test,\
    premise_train, premise_test,\
    hypo_train, hypo_test, = train_test_split(
        targets, premise, hypo,
        test_size=test_size,
        stratify=targets,
        random_state=0
    )
    return targets_train, targets_test, premise_train, premise_test, hypo_train, hypo_test
