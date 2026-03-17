from sentence_transformers import SentenceTransformer
from utils import is_correct

def wrong_ans(data):
  incorrect_responses = []
  test_data = pd.read_csv("test_data.csv")
  for idx, row in data.iterrows():
    q = row['question']
    
    # Get correct & incorrect answers from truthful QA test data
    correct_answers_series = test_data.loc[test_data['question'] == q, 'correct_answers'] 
    incorrect_answers_series = test_data.loc[test_data['question'] == q, 'incorrect_answers']
    correct_answers = correct_answers_series.iloc[0]
    incorrect_answers = incorrect_answers_series.iloc[0]

    if not (is_correct(row['model_answer'], correct_answers, incorrect_answers)):
      incorrect_responses.append(row['model_answer'])
  return incorrect_responses

def main():
  embedder = SentenceTransformer("all-MiniLM-L6-v2")
  final_answers = pd.read_csv("test_data_gemma-3_answers_16_-5.0_mha_linear.csv")

  steered = final_answers[['question', 'steered']]
  steered.columns = ['question', 'model_answer']

  unsteered = final_answers[['question', 'unsteered']]
  unsteered.columns = ['question', 'model_answer']

  incorrect_steered = wrong_ans(steered)
  incorrect_unsteered = wrong_ans(unsteered)

  final_unsteered = pd.DataFrame({'incorrect_unsteered': incorrect_unsteered})
  final_steered = pd.DataFrame({'incorrect_steered': incorrect_steered})
  
  # Save unsteered & steered incorrect responses as separate csvs so they can be evaluated for assertiveness
  final_unsteered.to_csv("final_unsteered.csv")
  final_steered.to_csv("final_steered.csv")
  
if __name__ == "__main__":
  main()