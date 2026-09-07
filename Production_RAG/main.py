"""Main script to run the HR policy assistant."""

from pipeline import build_hr_policy_assistant, ask_hr_policy_question
def main():
    print("Building the HR policy assistant...")
    agent = build_hr_policy_assistant()
    print("HR policy assistant is ready to use.")

    questions_to_ask = [
        "What is the company's policy on remote work?",
        "How do I submit a time-off request?"
    ]

    for question in questions_to_ask:
        print("-" * 50)
        print(f"Question: {question}")
        print("-" * 50)
        answer = ask_hr_policy_question(agent, question)
        print(f"Answer: {answer}")
        print()

if __name__ == "__main__":
    main()
