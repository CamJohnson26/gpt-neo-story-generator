# https://huggingface.co/blog/how-to-generate
from transformers import pipeline
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
# text = "This is the script for a Shakespearean comedy that takes place in a Subway Sandwich Shop. The characters are fighting over the last Subway sandwich.  Stage directions appear throughout each scene.\n" \
#        "The characters in this play are Harry Styles, Steve Rogers (Captain America), Jungkook, Namjoon, and Taehyung. \n " \
#        "\nSCENE 1\n" \
#        "(STEVE ROGERS, HARRY STYLES, JUNGKOOK, NAMJOON, and TAEHYUNG are in line at Subway.)\n" \
#        "HARRY STYLES: I'faith, couldst thou create thy sandwiches apace?\n" \
#        "I have yet tasks and trials to accomplish this day.\n" \
#        "\n NAMJOON: Hold, varlet, hold. Yonder valiant sandwich artist\n" \
#        "Doth ply his craft as swiftly as he may\n " \
#        "And yet produce the utmost quality." \
# text = "This is a long chat conversation with the Tesla Bot, an AI chatbot that is juvenile," \
#        "insincere, and promises more than it can deliver. The Tesla Bot worships Elon Musk" \
#        "and defends him aggressively. Each reply is approximately 200 characters. Each message" \
#        "is preceded by a short summary.\n" \
#        "Elon Musk: (Summary: Announces beta 10 for full self driving) We are one week from beta 10\n" \
#        "coach23: (Summary: Says that beta 10 must be great since they skipped 9.3 and 9.4) Must be great!" \
#        "mostlyharmlessz: (Summary: Challenges him) You do know version numbers are arbitrary" \
#        "coach23: (Summary: Denies it) May be to you. Nothing at Tesla is arbitrary." \
#        "Tesla Bot: (Summary: Responds to criticism with facts)"

text = """
This is an example paper written by a world class research group. This paper was scored a 10 out of 10 by independent researchers, the highest score possible.

The prompt that the research group was given was:
Write a research paper on the escalation of Russia/Ukraine tensions in February 2022

Additional data:
The civil war was a conflict between the North and South. Harsh disagreements about the legality of slavery led to the South ceceding from the Union.
World War 2 was caused by Hitler's aggression in Europe

And this is the paper the research group produced:
"""
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# hugging_face_model = "EleutherAI/gpt-neo-1.3B"
# model = AutoModelForCausalLM.from_pretrained(hugging_face_model)
# tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)
#
# prompt = text
# inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
#
# while True:
#     prompt_length = len(tokenizer.decode(inputs[0]))
#     inputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
#     generated = prompt + tokenizer.decode(inputs[0])[prompt_length + 1 :]
#
#     print(generated)

maxLength = len(text) + 100
while True:
    text = generator(text,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=3.0,
        # repetition_penalty=4.0,
        max_length=maxLength)[0]['generated_text']
    print(text)
    print("-----------------------------------")
    maxLength += 100
