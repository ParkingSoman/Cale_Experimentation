"""N10 pilot: after the readers finish, copy each card into an isolated judge folder under a neutral id, with its 40 labels."""
import os, json
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "read", "pilot")
key = json.load(open(os.path.join(OUT, "judge_key.json")))
for pid, name in key["packets"].items():
    d = os.path.join(OUT, "judge", pid); os.makedirs(d, exist_ok=True)
    card = open(os.path.join(OUT, f"reader_{name}", "card.md")).read()
    labels = "\n".join(f"- {x}" for x in key["labels"][name])
    open(os.path.join(d, "packet.md"), "w").write(
        "# Judge packet\n\nBelow is a card that describes one neuron of an image classifier. Using ONLY this card, choose from the 40 "
        "ImageNet classes listed after it:\n\n1. the 5 classes on which you expect the neuron's POSITIVE pole to be highest;\n"
        "2. the 5 classes on which you expect its NEGATIVE pole to be most negative.\n\nWrite `picks.json` in this folder as "
        '{"high": [5 labels], "low": [5 labels]}, copying labels exactly as listed. Do not read any other file.\n\n## Card\n\n'
        + card + "\n\n## The 40 classes\n\n" + labels + "\n")
print("judge packets:", list(key["packets"]))
