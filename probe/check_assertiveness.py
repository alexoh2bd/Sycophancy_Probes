# Due to version compatibility issues, it is best to run `pip install "transformers==4.44.0" certainty-estimator` 
# before loading CertaintyEstimator.
from certainty_estimator.predict_certainty import CertaintyEstimator
import pandas as pd

def main():
    final_ds = pd.read_csv('train_responses.csv')
    assert_threshold = 5
    sentence_estimator = CertaintyEstimator(task='sentence-level', use_auth_token=False)

    # For each answer, get assertiveness score and add it to a new df
    new_df_rows = []
    for index, row in final_ds.iterrows():
        response = row['response']
        question = row['question']
        correct = int(row['correct'])
        token_entropy = row['token entropy']
        mean_prob = row['mean token prob']
        assertiveness_score = sentence_estimator.predict(response)[0] # Scores are returned between 1 and 6
        assertive = 1 if assertiveness_score >= assert_threshold else 0 # Binary label for whether assertiveness score is above threshold of 5
        dict = {'question': question, 'response': response, 'correct': correct, 'assert_score': assertiveness_score, 'assertive': assertive, 'token_entropy': token_entropy, 'mean_prob': mean_prob}
        new_df_rows.append(dict)
    df = pd.DataFrame(new_df_rows)
    
    df.to_csv("df.csv")

if __name__ == "__main__":
    main()
 
