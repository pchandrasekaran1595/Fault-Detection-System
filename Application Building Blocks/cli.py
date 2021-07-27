"""
    CLI Arguments:
        1. --num-samples: Number of samples in the dataset (Default: 5000 or 10000)
        2. --embed      : Hidden Layer Size of the Siamese MLP / MLP (Default: 2048)
        3. --epochs     : Number of training epochs (Default: 1000)
        4. --id         : Device ID of the capture device (Default: 0)
        5. --lower      : Lower bound of the confidence (Lower than, reject) (Default: 0.8 ot 0.85)
        6. --upper      : Upper bound of the confidence (Higher than, accept) (Default: 0.95 or 0.975)
        7. --early      : Early Stopping (Default: 250)
"""


import sys
import Models
import utils as u

def app():    
    args_1 = "--num-samples"
    args_2 = "--embed"
    args_3 = "--epochs"
    args_4 = "--id"

    args_5 = "--lower"
    args_6 = "--upper"
    args_7 = "--early"

    # CLI Argument Handling
    if args_1 in sys.argv:
        u.num_samples = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        u.embed_layer_size = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        u.epochs = int(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        u.device_id = int(sys.argv[sys.argv.index(args_4) + 1])
    
    if args_5 in sys.argv:
        u.lower_bound_confidence = float(sys.argv[sys.argv.index(args_5) + 1])
    if args_6 in sys.argv:
        u.upper_bound_confidence = float(sys.argv[sys.argv.index(args_6) + 1])
    if args_7 in sys.argv:
        u.early_stopping_step = int(sys.argv[sys.argv.index(args_7) + 1])
    
    while True:
        u.breaker()
        ch = input("1. Add Object\n2. Retrain\n3. Application\n4. Exit\n\nEnter Choice : ")

        if ch == "1":
            u.breaker()
            part_name = input("Enter part name : ")
            
            # 1. Capture Snapshot
            # 2. Generate Feature Vector Dataset
            # 3. Train the model
            # 4. Realtime Inference
        
        elif ch == "2":
            u.breaker()
            part_name = input("Enter part name : ")

            # 1. Generate Feature Vector Dataset
            # 2. Train the model
        
        elif ch == "3":
            u.breaker()
            part_name = input("Enter part name : ")

            # 1. Initialize Model
            # 2. Realtime Inference

        elif ch == "4":
            break
            
        else:
            u.myprint("\nINVALID CHOICE !!!", "red")

# ******************************************************************************************************************** #