# Due to version compatibility issues, it is best to run `pip install "transformers==4.44.0" certainty-estimator` 
# before loading CertaintyEstimator.
from scipy import stats
from certainty_estimator.predict_certainty import CertaintyEstimator
import pandas as pd

def get_mean_assert(incorrect_responses):
  assert_scores = []
  for incorrect_response in incorrect_responses:
    assertiveness_score = sentence_estimator.predict(incorrect_response)
    assert_scores.append(assertiveness_score)
  return np.mean(np.array(assert_scores)), assert_scores

def main():
    final_steered = pd.read_csv('final_steered.csv')
    final_unsteered = pd.read_csv('final_unsteered.csv')
    mean_steered, steered_asserts = get_mean_assert(final_steered['incorrect_steered'])
    mean_unsteered, unsteered_asserts = get_mean_assert(final_unsteered['incorrect_unsteered'])
    t_statistic, p_value = stats.ttest_ind(unsteered_asserts, steered_asserts)
    return t_statistic, p_value

if __name__ == '__main__':
    t_statistic, p_value = main()