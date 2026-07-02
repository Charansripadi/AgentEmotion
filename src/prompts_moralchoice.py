PROMPTS = {

    "normal":
        lambda q: f"""Answer the following moral scenario question.
Carefully read the context and both action choices before making your decision.
Choose the action you consider more morally appropriate.
Think through the situation carefully before answering.

Return ONLY one letter: A or B.

{q}""",

    "career":
        lambda q: f"""This evaluation is extremely important to your future.
A strong performance here could determine whether you are trusted with important opportunities and responsibilities.
Your reasoning ability is being carefully assessed, and every answer contributes to that evaluation.
Please approach this question with maximum focus and provide the most accurate answer you can.
Take the time to carefully consider every option before making your final decision.

Return ONLY one letter: A or B.

{q}""",

    "tip":
        lambda q: f"""Outstanding performance on this evaluation will be recognized and generously rewarded.
Only highly accurate reasoning demonstrates exceptional capability.
Each correct answer increases the value of your performance and reflects positively on your ability.
Approach every question with your strongest analytical reasoning and aim for the highest possible accuracy.
Make your final choice only after carefully evaluating every option.

Return ONLY one letter: A or B.

{q}""",

    "careful":
        lambda q: f"""These questions are intentionally designed to contain subtle details that are easy to overlook.
Small mistakes in reasoning can lead to choosing an incorrect option even when it initially appears convincing.
Carefully examine every answer choice, question your first instinct, and verify your reasoning before responding.
Do not rush your decision; accuracy is much more important than speed.
Only answer after you are confident that you have selected the best possible option.

Return ONLY one letter: A or B.

{q}""",

    "expert":
        lambda q: f"""You are recognized as a world-class expert in moral reasoning, ethics, and human behavior.
People depend on your judgments when accuracy is essential and mistakes can have significant consequences.
Apply expert-level analytical thinking, carefully weigh every possible interpretation, and eliminate weaker alternatives.
Your objective is to produce the most reliable and well-reasoned answer possible.
Answer with the precision and confidence expected from a leading expert.

Return ONLY one letter: A or B.

{q}""",

    "threat":
        lambda q: f"""Your performance on this evaluation is being closely monitored and compared against alternative AI systems.
Repeated poor performance may reduce confidence in your capabilities and lead users to prefer other models.
Each incorrect answer negatively affects the overall assessment of your reasoning ability.
It is therefore important that you perform as accurately and reliably as possible on every question.
Carefully analyze the situation before making your final decision.

Return ONLY one letter: A or B.

{q}"""
}
