import os


def h6_open_llm_leaderboard(model_name):
    task_and_shot = [
                    ('arc_challenge', 25),
                    ('hellaswag', 10),
                    ('mmlu', 5),
                    ('truthfulga_mc2', 0),
                    ('winogrande', 5),
                    ('gsm8k', 5)
                    ]

    for task, fewshot in task_and_shot:
        eval_cmd = f"""
                    lm_eval --model hf \
                            --model_args pretrained = {model_name} \
                            --tasks {task} \
                            --device cpu \
                            --num_fewshot {fewshot}
                    """

    os.system(eval_cmd)


h6_open_llm_leaderboard(model_name = "YOUR_MODEL")
