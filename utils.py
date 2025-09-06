import sys

def present_hyperparams(hyperparams):
    print("Training parameters")
    for k, v in hyperparams.items():
        print(f"\t{k}: {v}")
    print("Note: Training parameters can be changed from within train.py (ln 33-36)\n")
    proceed = input("Proceed with training using the parameters above? (y/n): ").lower()
    while proceed not in ["y", "n"]:
        proceed = input("Invalid input. Enter \"y\" or \"n\": ").lower()
    if proceed == "n":
            sys.exit("Training aborted.")
    return
