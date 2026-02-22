def parse_answer(text):
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return text.strip().split("\n")[-1].strip()
