import torch

from transformers import BertForQuestionAnswering, BertTokenizerFast
from scipy.special import softmax
# import plotly.express as px
import pandas as pd
import numpy as np

model_name= "deepset/bert-base-cased-squad2"

tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def predict_answer(context, question):
  inputs = tokenizer(question, context, return_tensors="pt")

  with torch.no_grad():
    outputs = model(**inputs)

  start_scores, end_scores = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]

  # scores_df = pd.DataFrame({
#     "Token Position": list(range(len(start_scores))) * 2,
#     "Score": start_scores.tolist() + end_scores.tolist(),
#     "Score Type": ["Start"] * len(start_scores) + ["End"] * len(end_scores)
# })

  start_idx = np.argmax(start_scores)
  end_idx = np.argmax(end_scores)

  confidence_score = (start_scores[start_idx] + end_scores[end_idx]) /2

  answer_ids = inputs.input_ids[0][start_idx: end_idx + 1]
  answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
  answer = tokenizer.convert_tokens_to_string(answer_tokens)

  if answer != tokenizer.cls_token:
    return answer, confidence_score
  else:
    return None, confidence_score

context = """
Real Madrid Club de Fútbol, commonly referred to as Real Madrid, is a Spanish professional football club based in Madrid. The club competes in La Liga, the top tier of Spanish football.

Founded in 1902 as Madrid Football Club, the club has traditionally worn a white home kit since its inception. The honorific title real is Spanish for "royal" and was bestowed to the club by King Alfonso XIII in 1920 together with the royal crown in the emblem. Real Madrid have played their home matches in the 83,186-capacity Santiago Bernabéu in central Madrid since 1947. Unlike most European sporting entities, Real Madrid's members (socios) have owned and operated the club throughout its history. The official Madrid anthem is the "Hala Madrid y nada más", written by RedOne and Manuel Jabois.[6] The club is one of the most widely supported in the world, and is the most followed football club on social media according to the CIES Football Observatory as of 2023[7][8] and was estimated to be worth $6.07 billion in 2023, making it the world's most valuable football club.[9] In 2024, it was the highest-earning football club in the world, with an annual revenue of €831.4 million.[10]

In domestic football, the club has won 70 trophies; a record 35 La Liga titles, 20 Copa del Rey, 13 Supercopa de España, a Copa Eva Duarte and a Copa de la Liga.[11] In International football, Real Madrid have won a record 31 trophies: a record 14 European Cup/UEFA Champions League titles, a joint record five UEFA Super Cups, two UEFA Cups, a joint record two Latin Cups and a record eight FIFA Club World championships.[note 1] Madrid was ranked first in the International Federation of Football History & Statistics Club World Ranking for 2000, 2002, 2014, 2017.[15] In UEFA, Madrid ranks first in the all-time club ranking.[16][17]

Being one of the three founding members of La Liga that have never been relegated from the top division since its inception in 1929 (along with Athletic Bilbao and Barcelona), Real Madrid has many long-standing rivalries, most notably El Clásico with Barcelona and El Derbi Madrileño with Atlético Madrid. The club established itself as a major force in both Spanish and European football during the 1950s and 60s, winning five consecutive and six overall European Cups and reaching a further two finals. This success was replicated on the domestic front, with Madrid winning 12 league titles in 16 years. This team, which included Alfredo Di Stéfano, Ferenc Puskás, Paco Gento and Raymond Kopa is considered, by some in the sport, to be the greatest of all time.[18][19] Real Madrid is known for its Galácticos policy, which involves signing the world's best players, such as Ronaldo, Zidane and David Beckham to create a superstar team.[20] The term 'Galácticos policy' generally refers to the two eras of Florentino Pérez's presidency of the club (2000–2006 and 2009–2018), however, players brought in just before his tenure are sometimes considered to be part of the Galácticos legacy. A notable example is Steve McManaman, who like many other players also succeeded under the policy.[21] On 26 June 2009, Madrid signed Cristiano Ronaldo for a record breaking £80 million (€94 million);[22] he became both the club and history's all-time top goalscorer.[23][24][25][26] Madrid have recently relaxed the Galácticos policy, instead focusing on signing young talents such as Vinícius Júnior, Rodrygo and Jude Bellingham
"""


sentences = context.split("\n")
print(f"Length of sentences: {len(sentences)}")

def chunk_sentences(sentences, chunk_size, stride):
  chunks =[]
  num_sentences = len(sentences)

  for i in range(0, num_sentences, chunk_size - stride):
    chunk = sentences[i: i + chunk_size]
    chunks.append(chunk)

  return chunks

chunked_sentences = chunk_sentences(sentences, chunk_size=3, stride=1)

questions = ["What is coffee?", "What are the most common coffee beans?", "How many people are dependent on coffee for their income?"]

response = {}

while True:
    prompt = input("Prompt: ")
    if prompt == "exit":
        break
    for chunk in chunked_sentences:
      context = "\n".join(chunk)
      answer, score = predict_answer(context, prompt)
      if answer:
        if prompt not in response:
          response[prompt] = (answer, score)
        else:
          if score > response[prompt][1]:
            response[prompt] = (answer, score)
    print("Response:", response)
    response = {}

# for chunk in chunked_sentences:
#   context = "\n".join(chunk)
#   for question in questions:
#     answer, score = predict_answer(context, question)

#     if answer:
#       if question not in answers:
#         answers[question] = (answer, score)
#       else:
#         if score > answers[question][1]:
#           answers[question] = (answer, score)

# print(answers)
