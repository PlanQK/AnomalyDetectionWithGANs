import pandas as pd
import sklearn.model_selection
import sys

errorMsg = """
Usage: preprocessData.py inputfile
Arguments:
    inputfile: input data that needs preprocessing
               (split into test & train + normalization)
"""


def main():
    assert len(sys.argv) == 2, errorMsg
    creditFraudData = pd.read_csv(sys.argv[1])
    creditFraudData = creditFraudData.drop(["Time"], axis=1)

    # Normalize the input: This is especially important for SVMs
    def normalize(df, excludes=[]):
        result = df.copy()
        for feature_name in df.columns:
            if feature_name in excludes:
                continue
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (
                max_value - min_value
            )
        return result

    creditFraudData = normalize(creditFraudData, excludes=["Class"])

    training, testing = sklearn.model_selection.train_test_split(
        creditFraudData, test_size=0.10, random_state=42
    )

    # to make sure the testset is not to large
    testing = testing[:200]
    training.to_csv("input-data/trainSet.csv", index=False)
    testing.to_csv("input-data/predictionSet.csv", index=False)


if __name__ == "__main__":
    main()
