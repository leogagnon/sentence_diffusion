from transformers import pipeline
from tasks.autoencoder import AETask
import os
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer


os.environ['LATENT_CONTROL_CKPT_DIR'] = '/network/scratch/l/leo.gagnon/sentence_diffusion/logs/checkpoints/'


df = pd.read_csv("data/stories.csv")

sentences = [df[f"sentence{i}"].tolist() for i in range(1, 6)]
sentences = [
    " ".join([sentences[j][i] for j in range(5)])
    for i in range(len(sentences[0]))
]

topics = {
    # setting
    "at_home": "The story takes place at home.",
    "at_school": "The story takes place at school.",
    "at_workplace": "The story takes place at a workplace.",
    "at_hospital": "The story takes place in a hospital or clinic.",
    "at_store": "The story takes place in a store or mall.",
    "at_restaurant": "The story takes place at a restaurant or café.",
    "outdoor_city": "The story takes place outdoors in a city.",
    "outdoor_nature": "The story takes place outdoors in nature.",
    "at_sports_venue": "The story takes place at a sports venue or gym.",
    "during_travel": "The story takes place during travel or commuting.",
    "at_party": "The story takes place at a party or social gathering.",
    "online": "The story takes place online or via messages.",
    "bad_weather": "The story takes place in bad weather.",
    "time_morning": "The story happens in the morning.",
    "time_afternoon": "The story happens in the afternoon.",
    "time_evening": "The story happens in the evening or night.",
    "time_holiday": "The story happens on a weekend or holiday.",

    # protagonist
    "protagonist_man": "The protagonist is a man.",
    "protagonist_woman": "The protagonist is a woman.",
    "protagonist_child": "The protagonist is a child or teenager.",
    "protagonist_student": "The protagonist is a student.",
    "protagonist_worker": "The protagonist is an employee or worker.",
    "protagonist_parent": "The protagonist is a parent or caregiver.",
    "protagonist_friend": "The protagonist is a friend of another main character.",
    "protagonist_pet_owner": "The protagonist is a pet owner or deals with an animal.",
    "protagonist_athlete": "The protagonist is an athlete or hobbyist in sports.",
    "protagonist_customer": "The protagonist is a customer or client.",
    "protagonist_traveler": "The protagonist is a traveler or commuter.",

    # concrete themes
    "theme_school": "The story is about school or studying.",
    "theme_work": "The story is about work or a job.",
    "theme_sports": "The story is about sports or exercise.",
    "theme_health": "The story is about health or illness.",
    "theme_money": "The story is about money or budgeting.",
    "theme_shopping": "The story is about shopping or buying something.",
    "theme_food": "The story is about cooking or food.",
    "theme_pets": "The story is about pets or animals.",
    "theme_transport": "The story is about transportation or commuting.",
    "theme_tech": "The story is about technology or gadgets.",
    "theme_social_media": "The story is about social media or texting.",
    "theme_friendship": "The story is about friendship.",
    "theme_family": "The story is about family.",
    "theme_romance": "The story is about romance or dating.",
    "theme_celebration": "The story is about a celebration or holiday.",
    "theme_travel": "The story is about travel or a trip.",
    "theme_home_repairs": "The story is about home repairs or chores.",
    "theme_crime": "The story is about crime, loss, or damage.",
    "theme_art": "The story is about art, music, or entertainment.",
    "theme_volunteering": "The story is about volunteering or helping others.",

    # goals and plans
    "goal_set": "The protagonist sets a clear goal.",
    "goal_plan": "The protagonist makes a plan.",
    "goal_follow_rules": "The protagonist follows instructions or rules.",
    "goal_break_rules": "The protagonist breaks a rule to reach a goal.",
    "goal_practice": "The protagonist practices to improve.",
    "goal_compete": "The protagonist competes against others.",
    "goal_negotiate": "The protagonist negotiates to get what they want.",
    "goal_seek_help": "The protagonist seeks help from another person.",
    "goal_teach": "The protagonist teaches or explains something.",
    "goal_apologize": "The protagonist apologizes to someone.",

    # conflicts
    "conflict_self_mistake": "The story’s main problem is caused by the protagonist’s mistake.",
    "conflict_other_person": "The story’s main problem is caused by another person’s actions.",
    "conflict_bad_luck": "The story’s main problem is caused by bad luck or chance.",
    "conflict_external": "The story’s main problem is caused by external conditions (e.g., weather, traffic).",
    "conflict_internal": "The conflict is mainly internal to the protagonist.",
    "conflict_interpersonal": "The conflict is mainly between the protagonist and another person.",
    "conflict_system": "The conflict is mainly against a system or rule.",
    "stakes_high": "The stakes in the conflict are high for the protagonist.",
    "stakes_low": "The stakes in the conflict are low for the protagonist.",

    # narrative dynamics
    "ending_good": "The story ends well.",
    "ending_bad": "The story ends badly.",
    "ending_ambiguous": "The story has an ambiguous or mixed ending.",
    "goal_success": "The protagonist achieves their goal.",
    "goal_failure": "The protagonist fails to achieve their goal.",
    "setback": "The protagonist suffers a setback.",
    "adaptation": "The protagonist adapts their plan after a setback.",
    "moral": "The protagonist learns a lesson or moral.",
    "misunderstanding": "A misunderstanding drives the conflict.",
    "misunderstanding_resolved": "The misunderstanding is resolved.",
    "sacrifice": "The protagonist makes a sacrifice.",
    "unexpected_help": "The protagonist receives unexpected help.",
    "twist": "A surprising twist changes the situation.",
    "plan_success": "The plan works exactly as intended.",
    "plan_backfire": "The plan backfires.",
    "solution_communication": "The problem is solved by honest communication.",
    "solution_perseverance": "The problem is solved by perseverance.",
    "solution_creativity": "The problem is solved by creativity.",
    "solution_luck": "The problem is solved by luck.",

    # social relations
    "helping": "The protagonist helps another person.",
    "helped": "Another person helps the protagonist.",
    "hurting": "The protagonist hurts or wrongs another person.",
    "hurt_by": "Someone tries to hurt the protagonist.",
    "forgive_other": "The protagonist forgives someone.",
    "forgive_protagonist": "Someone forgives the protagonist.",
    "lie": "The protagonist lies or hides the truth.",
    "truth_cost": "The protagonist tells the truth despite a cost.",
    "keep_promise": "The protagonist keeps a promise.",
    "break_promise": "The protagonist breaks a promise.",

    # emotions
    "feel_happy": "The protagonist feels happy or pleased.",
    "feel_sad": "The protagonist feels sad or disappointed.",
    "feel_angry": "The protagonist feels angry or annoyed.",
    "feel_afraid": "The protagonist feels afraid or anxious.",
    "feel_guilty": "The protagonist feels guilty or ashamed.",
    "feel_proud": "The protagonist feels proud of themselves.",
    "feel_jealous": "The protagonist feels jealous or envious.",
    "feel_relieved": "The protagonist feels relieved at the end.",

    # risk and effort
    "risk": "The protagonist takes a risk.",
    "effort": "The protagonist invests time and effort.",
    "spend_money": "The protagonist spends money.",
    "save_money": "The protagonist saves money.",
    "lose_important": "The protagonist loses something important.",
    "find_valuable": "The protagonist finds or gains something valuable.",

    # causality
    "success_skill": "Success is mainly due to the protagonist’s skill or effort.",
    "success_help": "Success is mainly due to help from others.",
    "success_luck": "Success is mainly due to luck or timing.",
    "failure_choices": "Failure is mainly due to the protagonist’s poor choices.",
    "failure_obstacles": "Failure is mainly due to external obstacles.",
    "failure_luck": "Failure is mainly due to bad luck."
}

# pose sequence as a NLI premise and label as a hypothesis
nli_model = AutoModelForSequenceClassification.from_pretrained('MoritzLaurer/deberta-v3-large-zeroshot-v2.0', device_map='cuda', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained('MoritzLaurer/deberta-v3-large-zeroshot-v2.0')

hypotheses = list(topics.values())
hypotheses_keys = list(topics.keys())
with torch.no_grad():
    labels = torch.zeros((len(sentences), len(hypotheses)), device='cuda', dtype=torch.bool)

    for i, story in tqdm(enumerate(sentences), total=len(sentences)):
        x = tokenizer.batch_encode_plus([('Story : ' + story, h) for h in hypotheses], return_tensors='pt', truncation='only_first', padding=True).to('cuda')
        logits = nli_model(x['input_ids'], attention_mask=x['attention_mask']).logits
        probs = logits.softmax(dim=1)
        preds = probs[:,0] - probs[:,1]
        labels[i] = (preds > 0.90)

torch.save(labels, "data/story_labels.pt")