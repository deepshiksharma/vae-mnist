import sys

def present_hyperparams(hyperparams):
    print("Training parameters")
    for k, v in hyperparams.items():
        print(f"\t{k}: {v}")
    print("Note: Parameters can be changed within train.py (ln 28-31)\n")
    
    proceed = input("Proceed to train model using the parameters above? (y/n): ").lower()
    while proceed not in ["y", "n"]:
        proceed = input("Invalid input. Enter \"y\" or \"n\": ").lower()
    if proceed == "n":
            sys.exit("Training aborted.")
    return
